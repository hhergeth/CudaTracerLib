#include "PPPMTracer.h"
#include <Kernel/TraceHelper.h>
#include <Kernel/TraceAlgorithms.h>
#include <Math/half.h>
#include <Base/Timer.h>
#include <Kernel/ParticleProcess.h>

namespace CudaTracerLib {

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
	unsigned int* numStoredSurface;
	unsigned int* numStoredVolume;
	int numSurfaceInteractions;

	CUDA_FUNC_IN PPPMPhotonParticleProcessHandler(Image& I, Sampler& r, unsigned int* nStoredSuface, unsigned int* nStoredVol)
		: img(I), rng(r), wasStoredSurface(false), wasStoredVolume(false), numStoredSurface(nStoredSuface), numStoredVolume(nStoredVol), numSurfaceInteractions(0)
	{
	}

	CUDA_FUNC_IN void handleEmission(const Spectrum& weight, const PositionSamplingRecord& pRec)
	{
	}

	CUDA_FUNC_IN void handleSurfaceInteraction(const Spectrum& weight, float accum_pdf, const Spectrum& f, float pdf, const NormalizedT<Ray>& r, const TraceResult& r2, BSDFSamplingRecord& bRec, bool lastBssrdf, bool lastDelta)
	{
		auto wo = lastBssrdf ? r.dir() : -r.dir();
		if (rng.randomFloat() < g_Parameters.probSurface && r2.getMat().bsdf.hasComponent(ESmooth) && dot(bRec.dg.sys.n, wo) > 0.0f)
		{
			auto ph = PPPMPhoton(weight, wo, bRec.dg.sys.n);
			Vec3u cell_idx = g_SurfaceMap->getHashGrid().Transform(bRec.dg.P);
			ph.setPos(g_SurfaceMap->getHashGrid(), cell_idx, bRec.dg.P);
			bool b = false;
#ifdef ISCUDA
			if ((g_Parameters.DIRECT && numSurfaceInteractions > 0) || !g_Parameters.DIRECT)
			{
				auto idx = (g_Parameters.finalGathering && !lastDelta) || !g_Parameters.finalGathering ? g_SurfaceMap->Store(cell_idx, ph) : 0xffffffff;
				b |= idx != 0xffffffff;
				if (g_Parameters.finalGathering && lastDelta)
					b |= g_SurfaceMapCaustic->Store(cell_idx, ph) != 0xffffffff;
			}
#endif
			if (b && !wasStoredSurface)
			{
#ifdef ISCUDA
				atomicInc(numStoredSurface, UINT_MAX);
#endif
				wasStoredSurface = true;
			}
		}
		numSurfaceInteractions++;
	}

	CUDA_FUNC_IN void handleMediumSampling(const Spectrum& weight, float accum_pdf, const NormalizedT<Ray>& r, const TraceResult& r2, const MediumSamplingRecord& mRec, bool sampleInMedium, const VolumeRegion* bssrdf, bool lastDelta)
	{
		if(rng.randomFloat() < g_Parameters.probVolume)
		{
			auto ph = Beam(r.ori(), r.dir(), r2.m_fDist, weight);
			auto idx = ((VolEstimator*)g_VolEstimator)->StoreBeam(ph);
			if(idx != 0xffffffff && !wasStoredVolume)
			{
#ifdef ISCUDA
				atomicInc(numStoredVolume, UINT_MAX);
#endif
				wasStoredVolume = true;
			}
		}
	}

	CUDA_FUNC_IN void handleMediumInteraction(const Spectrum& weight, float accum_pdf, const Spectrum& f, float pdf, MediumSamplingRecord& mRec, const NormalizedT<Vec3f>& wi, const TraceResult& r2, const VolumeRegion* bssrdf, bool lastDelta)
	{
		if (rng.randomFloat() < g_Parameters.probVolume)
		{
			auto ph = _VolumetricPhoton(mRec.p, wi, weight);
			auto idx = ((VolEstimator*)g_VolEstimator)->StorePhoton(ph, mRec.p);
			if (idx != 0xffffffff && !wasStoredVolume)
			{
#ifdef ISCUDA
				atomicInc(numStoredVolume, UINT_MAX);
#endif
				wasStoredVolume = true;
			}
		}

		//connection to camera as in particle tracing
		/*if (!bssrdf)
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
		auto photon_idx = blockIdx.x * local_Todo + local_idx;
		auto rng = g_SamplerData(photon_idx);
		auto process = PPPMPhotonParticleProcessHandler<VolEstimator>(I, rng, &numStoredSurface, &numStoredVolume);
		ParticleProcess(PPM_MaxRecursion, PPM_MaxRecursion, rng, process);
		g_SamplerData(rng, photon_idx);
	}

	__syncthreads();
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		atomicAdd(&g_NumPhotonEmittedSurface, numStoredSurface);
		atomicAdd(&g_NumPhotonEmittedVolume, numStoredVolume);
	}
}

void PPPMTracer::doPhotonPass(Image* I)
{
	bool finalGathering = m_sParameters.getValue(KEY_N_FG_Samples()) != 0;

	m_sSurfaceMap.ResetBuffer();
	if (finalGathering)
		m_sSurfaceMapCaustic->ResetBuffer();
	m_pVolumeEstimator->StartNewPassBase(m_uPassesDone);
	m_pVolumeEstimator->StartNewPass(m_pScene);
	ThrowCudaErrors(cudaMemcpyToSymbol(g_SurfaceMap, &m_sSurfaceMap, sizeof(m_sSurfaceMap)));
	if (finalGathering)
		ThrowCudaErrors(cudaMemcpyToSymbol(g_SurfaceMapCaustic, m_sSurfaceMapCaustic, sizeof(*m_sSurfaceMapCaustic)));
	ZeroSymbol(g_NumPhotonEmittedSurface);
	ZeroSymbol(g_NumPhotonEmittedVolume);
	ThrowCudaErrors(cudaMemcpyToSymbol(g_VolEstimator, m_pVolumeEstimator, m_pVolumeEstimator->getSize()));

	PPPMParameters para = g_ParametersHost = { m_useDirectLighting, finalGathering, m_fProbSurface, m_fProbVolume, m_uPassesDone };
	para.DIRECT = g_ParametersHost.DIRECT;
	ThrowCudaErrors(cudaMemcpyToSymbol(g_ParametersDevice, &para, sizeof(para)));

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