#include "k_sPpmTracer.h"
#include "..\Kernel\k_TraceHelper.h"
#include "..\Kernel\k_TraceAlgorithms.h"

CUDA_DEVICE k_PhotonMapCollection g_Map;

CUDA_ONLY_FUNC bool storePhoton(const float3& p, const Spectrum& phi, const float3& wi, const float3& n, PhotonType type)
{
	unsigned int p_idx = atomicInc(&g_Map.m_uPhotonNumStored, 0xffffffff);
	if (p_idx < g_Map.m_uPhotonBufferLength)
	{
		g_Map.m_pPhotons[p_idx] = k_pPpmPhoton(p, phi, wi, n, type);
		//if this photon is caustic we will also have to store it in the diffuse map
		if (type == PhotonType::pt_Caustic)
		{
			p_idx = atomicInc(&g_Map.m_uPhotonNumStored, 0xffffffff);
			if (p_idx < g_Map.m_uPhotonBufferLength)
				g_Map.m_pPhotons[p_idx] = k_pPpmPhoton(p, phi, wi, n, PhotonType::pt_Diffuse);
		}
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
		const e_KernelBSSRDF* bssrdf = 0;

		while (++depth < PPM_MaxRecursion && g_Map.m_uPhotonNumStored < g_Map.m_uPhotonBufferLength && !Le.isZero())
		{
			TraceResult r2 = k_TraceRay(r);
			float minT, maxT;
			if ((!bssrdf && V.HasVolumes() && V.IntersectP(r, 0, r2.m_fDist, &minT, &maxT) && V.sampleDistance(r, 0, r2.m_fDist, rng, mRec))
				|| (bssrdf && sampleDistanceHomogenous(r, 0, r2.m_fDist, rng.randomFloat(), mRec, bssrdf->sig_a, bssrdf->sigp_s)))
			{
				throughput *= mRec.sigmaS * mRec.transmittance / mRec.pdfSuccess;
				wasStored |= storePhoton(mRec.p, throughput * Le, -r.direction, make_float3(0, 0, 0), PhotonType::pt_Volume);
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
						wasStored |= storePhoton(dg.P, throughput * Le, wo, bRec.dg.sys.n, delta ? PhotonType::pt_Caustic : PhotonType::pt_Diffuse);
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
	}

	g_RNGData(rng);
}

__global__ void buildHashGrid()
{
	unsigned int idx = threadId;
	if (idx < g_Map.m_uPhotonBufferLength)
	{
		k_pPpmPhoton& e = g_Map.m_pPhotons[idx];
		const k_PhotonMap<k_HashGrid_Reg>& map = (&g_Map.m_sSurfaceMap)[e.getType()];
		unsigned int i = map.m_sHash.Hash(e.getPos());
		unsigned int k = atomicExch(map.m_pDeviceHashGrid + i, idx);
		e.setNext(k);
	}
}

/*__global__ void buildHashGridLinkedList(float a_Radius)
{
	const float r2 = a_Radius * a_Radius;
	unsigned int idx = threadId;
	if (idx < g_Map.m_uPhotonBufferLength)
	{
		k_pPpmPhoton& e = g_Map.m_pPhotons[idx];
		k_PhotonMap<k_HashGrid_Reg>& map = (&g_Map.m_sSurfaceMap)[e.getType()];
		if (e.getType() == PhotonType::pt_Caustic || e.getType() == PhotonType::pt_Diffuse)
		{
			Frame f = Frame(e.getNormal());
			f.t *= a_Radius;
			f.s *= a_Radius;
			f.n *= a_Radius;
			float3 a = -1.0f * f.t - f.s, b = f.t - f.s, c = -1.0f * f.t + f.s, d = f.t + f.s;
			float3 low = fminf(fminf(a, b), fminf(c, d)) + e.getPos(), high = fmaxf(fmaxf(a, b), fmaxf(c, d)) + e.getPos();
			uint3 lo = map.m_sHash.Transform(low), hi = map.m_sHash.Transform(high);
			for (unsigned int a = lo.x; a <= hi.x; a++)
			for (unsigned int b = lo.y; b <= hi.y; b++)
			for (unsigned int c = lo.z; c <= hi.z; c++)
			{
				unsigned int hash_idx = map.m_sHash.Hash(make_uint3(a, b, c));
				unsigned int list_idx = atomicInc(&map.m_uLinkedListUsed, 0xffffffff);
				if (list_idx < map.m_uLinkedListLength)
				{
					unsigned int prev_list_idx = atomicExch(map.m_pDeviceHashGrid + hash_idx, list_idx);
					map.m_pDeviceLinkedList[list_idx] = make_uint2(idx, prev_list_idx);
				}
				else printf("list_idx = %d, length = %d", list_idx, map.m_uLinkedListLength);
			}
		}
	}
}*/

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
	{
		buildHashGrid<< <m_sMaps.m_uPhotonBufferLength / (32 * 6) + 1, dim3(32, 6, 1) >> >();
		cudaMemcpyFromSymbol(&m_sMaps, g_Map, sizeof(k_PhotonMapCollection));
	}
}