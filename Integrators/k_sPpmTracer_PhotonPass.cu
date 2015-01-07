#include "k_sPpmTracer.h"
#include "..\Kernel\k_TraceHelper.h"
#include "..\Kernel\k_TraceAlgorithms.h"

CUDA_DEVICE k_PhotonMapCollection g_Map;

template<typename HASH> k_StoreResult k_PhotonMap<HASH>::StorePhoton(const float3& p, const Spectrum& l, const float3& wi, const float3& n, unsigned int* a_PhotonCounter) const
{
	if(!m_sHash.IsValidHash(p))
		return k_StoreResult::NotValid;
	uint3 i0 = m_sHash.Transform(p);
	unsigned int i = m_sHash.Hash(i0);
#ifdef ISCUDA
	unsigned int j = atomicInc(a_PhotonCounter, 0xffffffff);
#else
	unsigned int j = InterlockedIncrement(a_PhotonCounter);
#endif
	if (j < m_uMaxPhotonCount)
	{
#ifdef ISCUDA
		unsigned int k = atomicExch(m_pDeviceHashGrid + i, j);
#else
		unsigned int k = InterlockedExchange(m_pDeviceHashGrid + i, j);
#endif
		m_pDevicePhotons[j] = k_pPpmPhoton(p, l, wi, n, k, 0);//m_sHash.EncodePos(p, i0)
		return k_StoreResult::Success;
	}
	return k_StoreResult::Full;
}

CUDA_ONLY_FUNC bool storePhoton(const float3& p, const Spectrum& phi, const float3& wi, const float3& n, unsigned int type)
{
	unsigned int p_idx = atomicInc(&g_Map.m_uPhotonNumStored, 0xffffffff);
	if (p_idx < g_Map.m_uPhotonBufferLength)
	{
		g_Map.m_pPhotons[p_idx] = k_pPpmPhoton(p, phi, wi, n, 0xffffffff, type);
		return true;
	}
	else return false;
}

template<bool DIRECT> __global__ void k_PhotonPass()
{ 
	CudaRNG rng = g_RNGData();
	CUDA_SHARED unsigned int local_Counter;
	local_Counter = 0;
	unsigned int local_Todo = PPM_Photons_Per_Thread * blockDim.x * blockDim.y;

	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	e_KernelAggregateVolume& V = g_SceneData.m_sVolume;

	while (local_Counter < local_Todo && g_Map.m_uPhotonNumStored < g_Map.m_uPhotonBufferLength)
	{
		Ray r;
		const e_KernelLight* light;
		Spectrum Le = g_SceneData.sampleEmitterRay(r, light, rng.randomFloat2(), rng.randomFloat2()),
				 throughput(1.0f);
		int depth = -1;
		atomicInc(&local_Counter, (unsigned int)-1);
		bool wasStored = false;
		bool delta = false;
		MediumSamplingRecord mRec;
		bool medium = false;
		e_KernelBSSRDF* bssrdf = 0;

		while (++depth < PPM_MaxRecursion && g_Map.m_uPhotonNumStored < g_Map.m_uPhotonBufferLength && !Le.isZero())
		{
			TraceResult r2 = k_TraceRay(r);
			float minT, maxT;
			if ((!bssrdf && V.HasVolumes() && V.IntersectP(r, 0, r2.m_fDist, &minT, &maxT) && V.sampleDistance(r, 0, r2.m_fDist, rng, mRec))
				|| (bssrdf && sampleDistanceHomogenous(r, 0, r2.m_fDist, rng.randomFloat(), mRec, bssrdf->sig_a, bssrdf->sigp_s)))
			{
				throughput *= mRec.sigmaS * mRec.transmittance / mRec.pdfSuccess;
				wasStored |= storePhoton(mRec.p, throughput * Le, -r.direction, make_float3(0, 0, 0), 2);
				if (bssrdf)
					r.direction = Warp::squareToUniformSphere(rng.randomFloat2());
				else throughput *= V.Sample(mRec.p, -r.direction, rng, &r.direction);
				r.origin = mRec.p;
				delta = false;
				medium = true;
			}
			else if (!r2.hasHit())
				break;
			else
			{
				if (medium)
					throughput *= mRec.transmittance / mRec.pdfFailure;
				float3 wo = bssrdf ? r.direction : -r.direction;
				r2.getBsdfSample(-wo, r(r2.m_fDist), &bRec, &rng);
				if ((DIRECT && depth > 0) || !DIRECT)
					if (r2.getMat().bsdf.hasComponent(EDiffuse) && dot(bRec.dg.sys.n, wo) > 0.0f)
						wasStored |= storePhoton(dg.P, throughput * Le, wo, bRec.dg.sys.n, 1);
				Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
				throughput *= f;
				delta = bRec.sampledType & ETypeCombinations::EDelta;
				
				if (!bssrdf && r2.getMat().GetBSSRDF(bRec.dg, &bssrdf))
					bRec.wo.z *= -1.0f;
				else bssrdf = 0;
				r = Ray(bRec.dg.P, bRec.getOutgoing());
			}
			if (depth > 3)
			{
				float q = MIN(throughput.max(), 0.95f);
				if (rng.randomFloat() >= q)
					break;
				throughput /= q;
			}
		}
		if (wasStored)
			atomicInc(&g_Map.m_uPhotonNumEmitted, 0xffffffff);
	restart_loop:;
	}

	g_RNGData(rng);
}

__global__ void buildHashGrid()
{
	unsigned int idx = threadId;
	if (idx < g_Map.m_uPhotonBufferLength)
	{
		k_pPpmPhoton& e = g_Map.m_pPhotons[idx];
		k_PhotonMap<k_HashGrid_Reg>& map = e.typeFlag == 1 ? g_Map.m_sSurfaceMap : g_Map.m_sVolumeMap;
		unsigned int i = map.m_sHash.Hash(e.getPos());
		unsigned int k = atomicExch(map.m_pDeviceHashGrid + i, idx);
		e.next = k;
	}
}

void k_sPpmTracer::doPhotonPass()
{
	cudaMemcpyToSymbol(g_Map, &m_sMaps, sizeof(k_PhotonMapCollection));
	k_INITIALIZE(m_pScene, g_sRngs);
	if(m_bDirect)
		k_PhotonPass<true> << < m_uBlocksPerLaunch, dim3(PPM_BlockX, PPM_BlockY, 1) >> >();
	else k_PhotonPass<false> << < m_uBlocksPerLaunch, dim3(PPM_BlockX, PPM_BlockY, 1) >> >();
	cudaThreadSynchronize();
	cudaMemcpyFromSymbol(&m_sMaps, g_Map, sizeof(k_PhotonMapCollection));
	if (m_sMaps.PassFinished())
		buildHashGrid << <m_sMaps.m_uPhotonBufferLength / (32 * 6) + 1, dim3(32, 6, 1) >> >();
}