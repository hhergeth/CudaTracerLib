#include "k_sPpmTracer.h"
#include "..\Kernel\k_TraceHelper.h"

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
		m_pDevicePhotons[j] = k_pPpmPhoton(p, l, wi, n, k);//m_sHash.EncodePos(p, i0)
		return k_StoreResult::Success;
	}
	return k_StoreResult::Full;
}

template<bool DIRECT> __global__ void k_PhotonPass()
{ 
	CudaRNG rng = g_RNGData();
	CUDA_SHARED unsigned int local_Counter;
	local_Counter = 0;
	__syncthreads();

	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	e_KernelAggregateVolume& V = g_SceneData.m_sVolume;
	TraceResult r2;
	Ray r;
	const e_KernelLight* light;
	Spectrum Le;

	unsigned int local_idx;
	while ((local_idx = atomicInc(&local_Counter, (unsigned int)-1)) < PPM_photons_per_block)
	{
		Le = g_SceneData.sampleEmitterRay(r, light, rng.randomFloat2(), rng.randomFloat2());
		r2.Init();
		int depth = -1;
		//bool inMesh = false;

		//__syncthreads();

		while (++depth < PPM_MaxRecursion && k_TraceRay(r.direction, r.origin, &r2))
		{
			/*if (V.HasVolumes())
			{
				float minT, maxT;
				while(V.IntersectP(r, 0, r2.m_fDist, &minT, &maxT))
				{
					float3 x = r(minT), w = -r.direction;
					Spectrum sigma_s = V.sigma_s(x, w), sigma_t = V.sigma_t(x, w);
					float d = -logf(rng.randomFloat()) / sigma_t.average();
					bool cancel = d >= (maxT - minT) || d >= r2.m_fDist;
					d = clamp(d, minT, maxT);
					Le += V.Lve(x, w) * d;
					if(g_Map.StorePhoton<false>(r(minT + d * rng.randomFloat()), Le, w, make_float3(0,0,0)) == k_StoreResult::Full)
						return;
					if(cancel)
						break;
					float A = (sigma_s / sigma_t).average();
					if(rng.randomFloat() <= A)
					{
						float3 wi;
						float pf = V.Sample(x, -r.direction, rng, &wi);
						Le /= A;
						Le *= pf;
						r.origin = r(minT + d);
						r.direction = wi;
						r2.Init();
						if(!k_TraceRay(r.direction, r.origin, &r2))
							goto restart_loop;
					}
					else break;//Absorption
				}
			}*/
			r2.getBsdfSample(r, rng, &bRec);
			Spectrum ac;
			/*const e_KernelBSSRDF* bssrdf;
			if(r2.getMat().GetBSSRDF(bRec.dg, &bssrdf))
			{
				//inMesh = false;
				ac = Le;
				Ray br = BSSRDF_Entry(bssrdf, bRec.dg.sys, dg.P, r.direction);
				while(true)
				{
					TraceResult r3 = k_TraceRay(br);
					Spectrum sigma_s = bssrdf->sigp_s, sigma_t = bssrdf->sigp_s + bssrdf->sig_a;
					float d = -logf(rng.randomFloat()) / sigma_t.average();
					bool cancel = d >= (r3.m_fDist);
					d = clamp(d, 0.0f, r3.m_fDist);
					if (g_Map.StorePhoton<false>(br(d * rng.randomFloat()), ac, -br.direction, make_float3(0, 0, 0)) == k_StoreResult::Full)
						return;
					if(cancel)
					{
						DifferentialGeometry dg2;
						r3.fillDG(dg2);
						dg.P = br(r3.m_fDist);//point on a surface
						Ray out = BSSRDF_Exit(bssrdf, dg2.sys, dg.P, br.direction);
						bRec.wo = bRec.dg.toLocal(out.direction);//ugly
						break;
					}
					float A = (sigma_s / sigma_t).average();
					if(rng.randomFloat() <= A)
					{
						ac /= A;
						float3 wo = Warp::squareToUniformSphere(rng.randomFloat2());
						ac *= 1.f / (4.f * PI);
						br = Ray(br(d), wo);
					}
					else goto restart_loop;
				}
				dg.P = r(r2.m_fDist);
			}
			else*/
			{
				float3 wo = -r.direction;
				if((DIRECT && depth > 0) || !DIRECT)
				if (r2.getMat().bsdf.hasComponent(EDiffuse) && dot(bRec.dg.sys.n, wo) > 0.0f)
					//if (g_Map.StorePhoton<true>(dg.P, Le, wo, bRec.dg.sys.n) == k_StoreResult::Full)
					//		return;
				{
					uint3 i0 = g_Map.m_sSurfaceMap.m_sHash.Transform(dg.P);//PPM_photons_per_block
					unsigned int j = blockIdx.x * PPM_slots_per_block + local_idx * PPM_MaxRecursion + depth;
					unsigned int i = g_Map.m_sSurfaceMap.m_sHash.Hash(i0);
					unsigned int k = atomicExch(g_Map.m_sSurfaceMap.m_pDeviceHashGrid + i, j);
					if (k == j)
						printf("Fucked up hash grid, created loop, enjoy ! k : %d", k);
					g_Map.m_pPhotons[j] = k_pPpmPhoton(dg.P, Le, wo, bRec.dg.sys.n, k);
				}
				Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
				//inMesh = dot(r.direction, bRec.map.sys.n) < 0;
				ac = Le * f;
			}
			/*if(depth > 5)
			{
				float prob = MIN(1.0f, ac.max() / Le.max());
				if(rng.randomFloat() > prob)
					break;
				Le = ac / prob;
			}
			else */Le = ac;
			r = Ray(dg.P, bRec.getOutgoing());
			r2.Init();
		}
		atomicInc(&g_Map.m_uPhotonNumEmitted, 0xffffffff);
	restart_loop:;
	}

	g_RNGData(rng);
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
	m_sMaps.m_uPhotonNumStored = m_sMaps.m_uPhotonBufferLength;
}