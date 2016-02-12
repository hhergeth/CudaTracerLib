#include "PPPMTracer.h"
#include <Kernel/TraceHelper.h>
#include <Kernel/TraceAlgorithms.h>
#include <Math/half.h>
#include <Base/Timer.h>

namespace CudaTracerLib {

CUDA_ONLY_FUNC bool BeamBeamGrid::StoreBeam(const Beam& b)
{
	unsigned int beam_idx = atomicInc(&m_uBeamIdx, (unsigned int)-1);
	if (beam_idx < m_uBeamLength)
	{
		m_pDeviceBeams[beam_idx] = b;
		bool storedAll = true;
#ifdef ISCUDA
		//const AABB objaabb = b.getAABB(m_fCurrentRadiusVol);
		//const int maxAxis = b.getDir().abs().arg_max();
		//const int chopCount = (int)(objaabb.Size()[maxAxis] * m_sStorage.getHashGrid().m_vInvSize[maxAxis]) + 1;
		//const float invChopCount = 1.0f / (float)chopCount;

		//for (int chop = 0; chop < chopCount; ++chop)
		{
			//AABB aabb = b.getSegmentAABB((chop)* invChopCount, (chop + 1) * invChopCount, m_fCurrentRadiusVol);

			//m_sStorage.ForAllCells(aabb.minV, aabb.maxV, [&](const Vec3u& pos)
			//{
				/*bool found_duplicate = false;
				m_sStorage.ForAll(pos, [&](unsigned int loc_idx, unsigned int b_idx)
				{
				if (found_duplicate) return;
				if (beam_idx == b_idx)
				found_duplicate = true;
				});
				if (!found_duplicate)*/
				//storedAll &= m_sStorage.store(pos, beam_idx);
			//});
		}

		//auto aabb = b.getAABB(m_fCurrentRadiusVol);
		//m_sStorage.ForAllCells(aabb.minV, aabb.maxV, [&](const Vec3u& pos)
		//{
		//	storedAll &= m_sStorage.store(pos, beam_idx);
		//});
#endif
		return storedAll;
	}
	else return false;
}

CUDA_CONST unsigned int g_PassIdx;
CUDA_DEVICE unsigned int g_NumPhotonEmittedSurface, g_NumPhotonEmittedVolume;
CUDA_DEVICE SurfaceMapT g_SurfaceMap;
CUDA_DEVICE SurfaceMapT g_SurfaceMapCaustic;
CUDA_DEVICE CUDA_ALIGN(16) unsigned char g_VolEstimator[Dmax4(sizeof(PointStorage), sizeof(BeamGrid), sizeof(BeamBeamGrid), sizeof(BeamBVHStorage))];

template<typename VolEstimator> __global__ void k_PhotonPass(int photons_per_thread, bool DIRECT, bool finalGathering, float probSurface, float probVolume)
{
	CudaRNG rng = g_RNGData();
	CUDA_SHARED unsigned int local_Counter;
	local_Counter = 0;
	unsigned int local_Todo = photons_per_thread * blockDim.x * blockDim.y;

	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	KernelAggregateVolume& V = g_SceneData.m_sVolume;
	CUDA_SHARED unsigned int numStoredSurface;
	CUDA_SHARED unsigned int numStoredVolume;
	numStoredSurface = 0; numStoredVolume = 0;
	__syncthreads();

	while (atomicInc(&local_Counter, (unsigned int)-1) < local_Todo && !g_SurfaceMap.isFull() && !((VolEstimator*)g_VolEstimator)->isFullK())
	{
		NormalizedT<Ray> r;
		const Light* light;
		Vec2f sps = rng.randomFloat2(), sds = rng.randomFloat2();
		Spectrum Le = g_SceneData.sampleEmitterRay(r, light, sps, sds),
			throughput(1.0f);
		int depth = -1;
		bool wasStoredSurface = false, wasStoredVolume = false;
		bool delta = false;
		MediumSamplingRecord mRec;
		bool medium = false;
		const VolumeRegion* bssrdf = 0;

		while (++depth < PPM_MaxRecursion && !Le.isZero() && !g_SurfaceMap.isFull() && !((VolEstimator*)g_VolEstimator)->isFullK())
		{
			TraceResult r2 = traceRay(r);
			float minT, maxT;
			bool inMedium = (!bssrdf && V.HasVolumes() && V.IntersectP(r, 0, r2.m_fDist, &minT, &maxT)) || bssrdf;
			bool storeVol = true;
			if (V.HasVolumes())
				storeVol = rng.randomFloat() < probVolume;
			if (inMedium && storeVol)
			{
				if (((VolEstimator*)g_VolEstimator)->StoreBeam(Beam(r.ori(), r.dir(), r2.m_fDist, throughput * Le)) && !wasStoredVolume)//store the beam even if sampled distance is too far ahead!
				{
					atomicInc(&numStoredVolume, UINT_MAX);
					wasStoredVolume = true;
				}
			}
			if ((!bssrdf && inMedium && V.sampleDistance(r, 0, r2.m_fDist, rng, mRec))
				|| (bssrdf && bssrdf->sampleDistance(r, 0, r2.m_fDist, rng.randomFloat(), mRec)))
			{//mRec.t
				throughput *= mRec.sigmaS * mRec.transmittance / mRec.pdfSuccess;
				if (storeVol)
					if (((VolEstimator*)g_VolEstimator)->StorePhoton(mRec.p, -r.dir(), throughput * Le) && !wasStoredVolume)
					{
						atomicInc(&numStoredVolume, UINT_MAX);
						wasStoredVolume = true;
					}
				if (bssrdf)
				{
					PhaseFunctionSamplingRecord pRec(-r.dir(), r.dir());
					throughput *= bssrdf->As()->Func.Sample(pRec, rng);
					r.dir() = pRec.wi;
				}
				else
				{
					NormalizedT<Vec3f> dir;
					throughput *= V.Sample(mRec.p, -r.dir(), rng, &dir);
					r.dir() = dir;
				}
				r.ori() = mRec.p;
				delta = false;
				medium = true;
			}
			else if (!r2.hasHit())
				break;
			else
			{
				if (medium)
					throughput *= mRec.transmittance / mRec.pdfFailure;
				auto wo = bssrdf ? r.dir() : -r.dir();
				Spectrum f_i = throughput * Le;
				r2.getBsdfSample(-wo, r(r2.m_fDist), bRec, ETransportMode::EImportance, &rng, &f_i);
				if (rng.randomFloat() < probSurface && r2.getMat().bsdf.hasComponent(ESmooth) && dot(bRec.dg.sys.n, wo) > 0.0f)
				{
					auto ph = PPPMPhoton(throughput * Le, wo, bRec.dg.n, delta ? PhotonType::pt_Caustic : PhotonType::pt_Diffuse);
					Vec3u cell_idx = g_SurfaceMap.getHashGrid().Transform(dg.P);
					ph.setPos(g_SurfaceMap.getHashGrid(), cell_idx, dg.P);
					bool b = false;
					if ((DIRECT && depth > 0) || !DIRECT)
					{
						b |= g_SurfaceMap.store(cell_idx, ph);
						if (finalGathering && delta)
							b |= g_SurfaceMapCaustic.store(cell_idx, ph);
					}
					if (b && !wasStoredSurface)
					{
						atomicInc(&numStoredSurface, UINT_MAX);
						wasStoredSurface = true;
					}
				}
				Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
				delta &= bRec.sampledType & ETypeCombinations::EDelta;
				if (!bssrdf && r2.getMat().GetBSSRDF(bRec.dg, &bssrdf))
					bRec.wo.z *= -1.0f;
				else
				{
					if (!bssrdf)
						throughput *= f;
					bssrdf = 0;
					medium = false;
				}

				r = NormalizedT<Ray>(bRec.dg.P, bRec.getOutgoing());
			}
		}
	}

	__syncthreads();
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		atomicAdd(&g_NumPhotonEmittedSurface, numStoredSurface);
		atomicAdd(&g_NumPhotonEmittedVolume, numStoredVolume);
	}

	g_RNGData(rng);
}

void PPPMTracer::doPhotonPass()
{
	bool finalGathering = m_sParameters.getValue(KEY_FinalGathering());

	m_sSurfaceMap.ResetBuffer();
	if (finalGathering)
		m_sSurfaceMapCaustic.ResetBuffer();
	m_pVolumeEstimator->StartNewPass(this, m_pScene);
	ThrowCudaErrors(cudaMemcpyToSymbol(g_SurfaceMap, &m_sSurfaceMap, sizeof(m_sSurfaceMap)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_SurfaceMapCaustic, &m_sSurfaceMapCaustic, sizeof(m_sSurfaceMapCaustic)));
	ZeroSymbol(g_NumPhotonEmittedSurface);
	ZeroSymbol(g_NumPhotonEmittedVolume);
	ThrowCudaErrors(cudaMemcpyToSymbol(g_VolEstimator, m_pVolumeEstimator, m_pVolumeEstimator->getSize()));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_PassIdx, &m_uPassesDone, sizeof(m_uPassesDone)));

	while (!m_sSurfaceMap.isFull() && !m_pVolumeEstimator->isFull())
	{
		if (dynamic_cast<BeamGrid*>(m_pVolumeEstimator))
			k_PhotonPass<BeamGrid> << < m_uBlocksPerLaunch, dim3(PPM_BlockX, PPM_BlockY, 1) >> >(PPM_Photons_Per_Thread, m_useDirectLighting, finalGathering, m_fProbSurface, m_fProbVolume);
		else if(dynamic_cast<PointStorage*>(m_pVolumeEstimator))
			k_PhotonPass<PointStorage> << < m_uBlocksPerLaunch, dim3(PPM_BlockX, PPM_BlockY, 1) >> >(PPM_Photons_Per_Thread, m_useDirectLighting, finalGathering, m_fProbSurface, m_fProbVolume);
		else if (dynamic_cast<BeamBeamGrid*>(m_pVolumeEstimator))
			k_PhotonPass<BeamBeamGrid> << < m_uBlocksPerLaunch, dim3(PPM_BlockX, PPM_BlockY, 1) >> >(PPM_Photons_Per_Thread, m_useDirectLighting, finalGathering, m_fProbSurface, m_fProbVolume);
		else if (dynamic_cast<BeamBVHStorage*>(m_pVolumeEstimator))
			k_PhotonPass<BeamBVHStorage> << < m_uBlocksPerLaunch, dim3(PPM_BlockX, PPM_BlockY, 1) >> >(PPM_Photons_Per_Thread, m_useDirectLighting, finalGathering, m_fProbSurface, m_fProbVolume);
		ThrowCudaErrors(cudaMemcpyFromSymbol(&m_sSurfaceMap, g_SurfaceMap, sizeof(m_sSurfaceMap)));
		ThrowCudaErrors(cudaMemcpyFromSymbol(&m_sSurfaceMapCaustic, g_SurfaceMapCaustic, sizeof(m_sSurfaceMapCaustic)));
		ThrowCudaErrors(cudaMemcpyFromSymbol(m_pVolumeEstimator, g_VolEstimator, m_pVolumeEstimator->getSize()));
	}
	ThrowCudaErrors(cudaMemcpyFromSymbol(&m_uPhotonEmittedPassSurface, g_NumPhotonEmittedSurface, sizeof(m_uPhotonEmittedPassSurface)));
	ThrowCudaErrors(cudaMemcpyFromSymbol(&m_uPhotonEmittedPassVolume, g_NumPhotonEmittedVolume, sizeof(m_uPhotonEmittedPassVolume)));
	m_pVolumeEstimator->PrepareForRendering();
	m_sSurfaceMap.PrepareForUse();
	if (finalGathering)
		m_sSurfaceMapCaustic.PrepareForUse();
	if (m_uTotalPhotonsEmitted == 0)
		doPerPixelRadiusEstimation();
	size_t volLength, volCount;
	m_pVolumeEstimator->getStatusInfo(volLength, volCount);
	if (m_adaptiveProbabilities)
	{
		m_fProbVolume = math::clamp01(m_fProbVolume * (float)m_sSurfaceMap.getNumStoredEntries() / m_sSurfaceMap.getNumEntries());
		m_fProbSurface = math::clamp01(m_fProbSurface * (float)volCount / volLength);
	}
}

}