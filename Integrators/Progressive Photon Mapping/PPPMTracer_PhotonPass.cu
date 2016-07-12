#include "PPPMTracer.h"
#include <Kernel/TraceHelper.h>
#include <Kernel/TraceAlgorithms.h>
#include <Math/half.h>
#include <Base/Timer.h>
#include <Kernel/ParticleProcess.h>

namespace CudaTracerLib {

CUDA_ONLY_FUNC bool BeamBeamGrid::StoreBeam(const Beam& b)
{
	unsigned int beam_idx = atomicInc(&m_uBeamIdx, (unsigned int)-1);
	if (beam_idx < m_sBeamStorage.getLength())
	{
		m_sBeamStorage[beam_idx] = b;
		bool storedAll = true;
#ifdef ISCUDA
		TraverseGridRay(Ray(b.pos, b.getDir()), m_sStorage.getHashGrid(), 0.0f, b.t, [&](float minT, float rayT, float maxT, float cellEndT, Vec3u& cell_pos, bool& cancelTraversal)
		{
			storedAll &= m_sStorage.store(cell_pos, beam_idx);
		});
#endif
		return storedAll;
	}
	else return false;
}

struct PPPMParameters
{
	bool DIRECT;
	bool finalGathering;
	float probSurface;
	float probVolume;
	unsigned int PassIdx;
};
CUDA_CONST PPPMParameters g_ParametersDevice;
static PPPMParameters g_ParametersHost;
#ifdef ISCUDA
#define g_Parameters g_ParametersDevice	
#else
#define g_Parameters g_ParametersHost
#endif
CUDA_DEVICE unsigned int g_NumPhotonEmittedSurface, g_NumPhotonEmittedVolume;
CUDA_DEVICE CudaStaticWrapper<SurfaceMapT> g_SurfaceMap;
CUDA_DEVICE CudaStaticWrapper<SurfaceMapT> g_SurfaceMapCaustic;
CUDA_DEVICE CUDA_ALIGN(16) unsigned char g_VolEstimator[Dmax3(sizeof(PointStorage), sizeof(BeamGrid), sizeof(BeamBeamGrid))];

template<typename VolEstimator> struct PPPMPhotonParticleProcessHandler
{
	Image& img;
	Sampler& rng;
	bool wasStoredSurface;
	bool wasStoredVolume;
	bool delta;
	unsigned int* numStoredSurface;
	unsigned int* numStoredVolume;
	int numSurfaceInteractions;

	CUDA_FUNC_IN PPPMPhotonParticleProcessHandler(Image& I, Sampler& r, unsigned int* nStoredSuface, unsigned int* nStoredVol)
		: img(I), rng(r), wasStoredSurface(false), wasStoredVolume(false), delta(true), numStoredSurface(nStoredSuface), numStoredVolume(nStoredVol), numSurfaceInteractions(0)
	{

	}

	CUDA_FUNC_IN void handleEmission(const Spectrum& weight, const PositionSamplingRecord& pRec)
	{

	}

	CUDA_FUNC_IN void handleSurfaceInteraction(const Spectrum& weight, const NormalizedT<Ray>& r, const TraceResult& r2, BSDFSamplingRecord& bRec, bool lastBssrdf)
	{
		auto wo = lastBssrdf ? r.dir() : -r.dir();
		if (rng.randomFloat() < g_Parameters.probSurface && r2.getMat().bsdf.hasComponent(ESmooth) && dot(bRec.dg.sys.n, wo) > 0.0f)
		{
			auto ph = PPPMPhoton(weight, wo, bRec.dg.n, delta ? PhotonType::pt_Caustic : PhotonType::pt_Diffuse);
			Vec3u cell_idx = g_SurfaceMap->getHashGrid().Transform(bRec.dg.P);
			ph.setPos(g_SurfaceMap->getHashGrid(), cell_idx, bRec.dg.P);
			bool b = false;
			if ((g_Parameters.DIRECT && numSurfaceInteractions > 0) || !g_Parameters.DIRECT)
			{
#ifdef ISCUDA
				b |= g_SurfaceMap->store(cell_idx, ph);
				if (g_Parameters.finalGathering && delta)
					b |= g_SurfaceMapCaustic->store(cell_idx, ph);
#endif
			}
			if (b && !wasStoredSurface)
			{
#ifdef ISCUDA
				atomicInc(numStoredSurface, UINT_MAX);
#endif
				wasStoredSurface = true;
			}
		}
		delta &= (bRec.sampledType & ETypeCombinations::EDelta) != 0;
		numSurfaceInteractions++;
	}

	template<bool BSSRDF> CUDA_FUNC_IN void handleMediumSampling(const Spectrum& weight, const NormalizedT<Ray>& r, const TraceResult& r2, const MediumSamplingRecord& mRec, bool sampleInMedium)
	{
		bool storeVol = rng.randomFloat() < g_Parameters.probVolume;
		if (storeVol && ((VolEstimator*)g_VolEstimator)->StoreBeam(Beam(r.ori(), r.dir(), r2.m_fDist, weight)) && !wasStoredVolume)
		{
#ifdef ISCUDA
			atomicInc(numStoredVolume, UINT_MAX);
#endif
			wasStoredVolume = true;
		}
	}

	template<bool BSSRDF> CUDA_FUNC_IN void handleMediumInteraction(const Spectrum& weight, MediumSamplingRecord& mRec, const NormalizedT<Vec3f>& wi, const TraceResult& r2)
	{
		delta = false;
		bool storeVol = rng.randomFloat() < g_Parameters.probVolume;
		if (storeVol && ((VolEstimator*)g_VolEstimator)->StorePhoton(mRec.p, wi, weight) && !wasStoredVolume)
		{
#ifdef ISCUDA
			atomicInc(numStoredVolume, UINT_MAX);
#endif
			wasStoredVolume = true;
		}

		//connection to camera as in particle tracing
		/*if (!BSSRDF)
		{
			DirectSamplingRecord dRec(mRec.p, NormalizedT<Vec3f>(0.0f));
			Spectrum value = weight * g_SceneData.sampleAttenuatedSensorDirect(dRec, rng.randomFloat2());
			if (!value.isZero() && V(dRec.p, dRec.ref))
			{
				PhaseFunctionSamplingRecord pRec(wi, dRec.d);
				value *= g_SceneData.m_sVolume.p(mRec.p, pRec);
				if (!value.isZero())
					img.Splat(dRec.uv.x, dRec.uv.y, value);
			}
		}*/
	}
};

template<typename VolEstimator> __global__ void k_PhotonPass(int photons_per_thread, Image I)
{
	auto rng = g_SamplerData();
	CUDA_SHARED unsigned int local_Counter;
	local_Counter = 0;
	unsigned int local_Todo = photons_per_thread * blockDim.x * blockDim.y;

	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	CUDA_SHARED unsigned int numStoredSurface;
	CUDA_SHARED unsigned int numStoredVolume;
	numStoredSurface = 0; numStoredVolume = 0;
	__syncthreads();

	unsigned int local_idx;
	while ((local_idx = atomicInc(&local_Counter, (unsigned int)-1)) < local_Todo && !g_SurfaceMap->isFull() && !((VolEstimator*)g_VolEstimator)->isFullK())
	{
		rng.StartSequence(blockIdx.x * local_Todo + local_idx);
		auto process = PPPMPhotonParticleProcessHandler<VolEstimator>(I, rng, &numStoredSurface, &numStoredVolume);
		ParticleProcess(PPM_MaxRecursion, PPM_MaxRecursion, rng, process);
	}

	__syncthreads();
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		atomicAdd(&g_NumPhotonEmittedSurface, numStoredSurface);
		atomicAdd(&g_NumPhotonEmittedVolume, numStoredVolume);
	}

	g_SamplerData(rng);
}

void PPPMTracer::doPhotonPass(Image* I)
{
	bool finalGathering = m_sParameters.getValue(KEY_FinalGathering());

	m_sSurfaceMap.ResetBuffer();
	if (finalGathering)
		m_sSurfaceMapCaustic->ResetBuffer();
	m_pVolumeEstimator->StartNewPass(this, m_pScene);
	ThrowCudaErrors(cudaMemcpyToSymbol(g_SurfaceMap, &m_sSurfaceMap, sizeof(m_sSurfaceMap)));
	if (finalGathering)
		ThrowCudaErrors(cudaMemcpyToSymbol(g_SurfaceMapCaustic, m_sSurfaceMapCaustic, sizeof(*m_sSurfaceMapCaustic)));
	ZeroSymbol(g_NumPhotonEmittedSurface);
	ZeroSymbol(g_NumPhotonEmittedVolume);
	ThrowCudaErrors(cudaMemcpyToSymbol(g_VolEstimator, m_pVolumeEstimator, m_pVolumeEstimator->getSize()));

	PPPMParameters para = g_ParametersHost = { m_useDirectLighting, finalGathering, m_fProbSurface, m_fProbVolume, m_uPassesDone };
	para.DIRECT = g_ParametersHost.DIRECT;
	ThrowCudaErrors(cudaMemcpyToSymbol(g_ParametersDevice, &para, sizeof(para)));

	setNumSequences(m_uBlocksPerLaunch * PPM_BlockX * PPM_BlockY * PPM_Photons_Per_Thread);

	while (!m_sSurfaceMap.isFull() && !m_pVolumeEstimator->isFull())
	{
		if (dynamic_cast<BeamGrid*>(m_pVolumeEstimator))
			k_PhotonPass<BeamGrid> << < m_uBlocksPerLaunch, dim3(PPM_BlockX, PPM_BlockY, 1) >> >(PPM_Photons_Per_Thread, *I);
		else if(dynamic_cast<PointStorage*>(m_pVolumeEstimator))
			k_PhotonPass<PointStorage> << < m_uBlocksPerLaunch, dim3(PPM_BlockX, PPM_BlockY, 1) >> >(PPM_Photons_Per_Thread, *I);
		else if (dynamic_cast<BeamBeamGrid*>(m_pVolumeEstimator))
			k_PhotonPass<BeamBeamGrid> << < m_uBlocksPerLaunch, dim3(PPM_BlockX, PPM_BlockY, 1) >> >(PPM_Photons_Per_Thread, *I);

		ThrowCudaErrors(cudaMemcpyFromSymbol(&m_sSurfaceMap, g_SurfaceMap, sizeof(m_sSurfaceMap)));
		if (finalGathering)
			ThrowCudaErrors(cudaMemcpyFromSymbol(m_sSurfaceMapCaustic, g_SurfaceMapCaustic, sizeof(*m_sSurfaceMapCaustic)));
		ThrowCudaErrors(cudaMemcpyFromSymbol(m_pVolumeEstimator, g_VolEstimator, m_pVolumeEstimator->getSize()));
	}
	ThrowCudaErrors(cudaMemcpyFromSymbol(&m_uPhotonEmittedPassSurface, g_NumPhotonEmittedSurface, sizeof(m_uPhotonEmittedPassSurface)));
	ThrowCudaErrors(cudaMemcpyFromSymbol(&m_uPhotonEmittedPassVolume, g_NumPhotonEmittedVolume, sizeof(m_uPhotonEmittedPassVolume)));
	m_sSurfaceMap.setOnGPU();
	if (finalGathering)
		m_sSurfaceMapCaustic->setOnGPU();
	m_pVolumeEstimator->setOnGPU();

	m_pVolumeEstimator->PrepareForRendering();
	m_sSurfaceMap.PrepareForUse();
	if (finalGathering)
		m_sSurfaceMapCaustic->PrepareForUse();
	if (m_uTotalPhotonsEmitted == 0)
		doPerPixelRadiusEstimation();
	size_t volLength, volCount;
	m_pVolumeEstimator->getStatusInfo(volLength, volCount);
	if (m_sParameters.getValue(KEY_AdaptiveAccProb()))
	{
		if (m_sSurfaceMap.getNumStoredEntries())
			m_fProbVolume = math::clamp01(m_fProbVolume * (float)m_sSurfaceMap.getNumStoredEntries() / m_sSurfaceMap.getNumEntries());
		if (volCount)
			m_fProbSurface = math::clamp01(m_fProbSurface * (float)volCount / volLength);
	}
}

}