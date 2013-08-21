#include "k_sPpmTracer.h"
#include "k_TraceHelper.h"
#include "..\Math\Montecarlo.h"

CUDA_DEVICE k_PhotonMapCollection g_Map;

template<bool DIRECT> CUDA_ONLY_FUNC bool TracePhoton(Ray& r, float3 Le, CudaRNG& rng)
{
	r.direction = normalize(r.direction);
	e_KernelAggregateVolume& V = g_SceneData.m_sVolume;
	TraceResult r2;
	r2.Init();
	int depth = -1;
	bool inMesh = false;
	e_KernelBSDF bsdf;
	while(++depth < 12 && k_TraceRay<true>(r.direction, r.origin, &r2))
	{
		if(V.HasVolumes())
		{
			float minT, maxT;
			while(V.IntersectP(r, 0, r2.m_fDist, &minT, &maxT))
			{
				float3 x = r(minT), w = -r.direction;
				float3 sigma_s = V.sigma_s(x, w), sigma_t = V.sigma_t(x, w);
				float d = -logf(rng.randomFloat()) / (fsumf(sigma_t) / 3.0f);
				bool cancel = d >= (maxT - minT) || d >= r2.m_fDist;
				d = clamp(d, minT, maxT);
				Le += V.Lve(x, w) * d;
				if(g_Map.StorePhoton<false>(r(minT + d * rng.randomFloat()), Le, w, make_float3(0,0,0)) == k_StoreResult::Full)
					return false;
				if(cancel)
					break;
				float A = fsumf(sigma_s / sigma_t) / 3.0f;
				if(rng.randomFloat() <= A)
				{
					float3 wi;
					float pf = V.Sample(x, -r.direction, rng, &wi);
					Le /= A;
					Le *= pf;
					r.origin = r(minT + d);
					r.direction = wi;
					r2.Init();
					if(!k_TraceRay<true>(r.direction, r.origin, &r2))
						return true;
				}
				else break;//Absorption
			}
		}
		float3 x = r(r2.m_fDist);
		e_KernelBSSRDF bssrdf;
		float3 ac, wi;
		if(r2.m_pTri->GetBSSRDF(x,r2.m_fUV, r2.m_pNode->getWorldMatrix(), g_SceneData.m_sMatData.Data, r2.m_pNode->m_uMaterialOffset, &bssrdf))
		{
			inMesh = false;
			ac = Le;
			while(true)
			{
				float3 w = -r.direction;
				TraceResult r3 = k_TraceRay<false>(Ray(x, r.direction));
				float3 sigma_s = bssrdf.sigp_s, sigma_t = bssrdf.sigp_s + bssrdf.sig_a;
				float d = -logf(rng.randomFloat()) / (fsumf(sigma_t) / 3.0f);
				bool cancel = d >= (r3.m_fDist);
				d = clamp(d, 0.0f, r3.m_fDist);
				if(g_Map.StorePhoton<false>(x + r.direction * (d * rng.randomFloat()), ac, w, make_float3(0,0,0)) == k_StoreResult::Full)
					return false;
				if(cancel)
				{
					x = x + r.direction * r3.m_fDist;
					wi = refract(r.direction, r3.m_pTri->lerpOnb(r3.m_fUV, r3.m_pNode->getWorldMatrix()).m_normal, bssrdf.e);
					break;
				}
				float A = fsumf(sigma_s / sigma_t) / 3.0f;
				if(rng.randomFloat() <= A)
				{
					ac /= A;
					float3 wo = Warp::squareToUniformSphere(rng.randomFloat2());
					ac *= 1.f / (4.f * PI);
					r.origin = x + r.direction * d;
					r.direction = wo;
				}
				else return true;
			}
		}
		else
		{
			r2.GetBSDF(x, &bsdf);
			float3 wo = -r.direction;
			if((DIRECT && depth > 0) || !DIRECT)
				if(bsdf.NumComponents(BxDFType(BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_DIFFUSE )))
					if(g_Map.StorePhoton<true>(x, Le, wo, bsdf.ng) == k_StoreResult::Full)
						return false;
			float pdf;
			BxDFType sampledType;
			float3 f = bsdf.Sample_f(wo, &wi, BSDFSample(rng), &pdf, BSDF_ALL, &sampledType);
			if(pdf == 0 || ISBLACK(f))
				break;
			inMesh = dot(r.direction, bsdf.ng) < 0;
			ac = Le * f * AbsDot(wi, bsdf.sys.m_normal) / pdf;
		}
		//if(depth > 3)
		{
			float prob = MIN(1.0f, fmaxf(ac) / fmaxf(Le));
			if(rng.randomFloat() > prob)
				break;
			Le = ac / prob;
		}
		//else Le = ac;
		r = Ray(x, normalize(wi));
		r2.Init();
	}
	return true;
}

template<bool DIRECT> __global__ void k_PhotonPass(unsigned int spp)
{ 
	const unsigned int a_MaxDepth = 10;
	CudaRNG rng = g_RNGData();
	for(int _photonNum = 0; _photonNum < spp; _photonNum++)
	{
		int li = (int)float(g_SceneData.m_sLightSelector.m_uCount) * rng.randomFloat();
		int li2 = g_SceneData.m_sLightSelector.m_sIndices[li];
		float lightPdf = 1.0f / (float)g_SceneData.m_sLightSelector.m_uCount;
	label001:
		Ray photonRay;
		float3 Nl;
		float pdf;
		float3 Le = g_SceneData.m_sLightData[li2].Sample_L(g_SceneData, LightSample(rng), rng.randomFloat(), rng.randomFloat(), &photonRay, &Nl, &pdf); 
		if(pdf == 0 || ISBLACK(Le))
			continue;
		float3 alpha = (AbsDot(Nl, photonRay.direction) * Le) / (pdf * lightPdf);
		if(TracePhoton<DIRECT>(photonRay, alpha, rng))
			atomicInc(&g_Map.m_uPhotonNumEmitted, -1);
		else break;
	}
	g_RNGData(rng);
}

void k_sPpmTracer::doPhotonPass()
{
	cudaMemcpyToSymbol(g_Map, &m_sMaps, sizeof(k_PhotonMapCollection));
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, g_sRngs);
	const unsigned long long p0 = 6 * 32, spp = 3, n = 180, PhotonsPerPass = p0 * n * spp;
	if(m_bDirect)
		k_PhotonPass<true><<< n, p0 >>>(spp);
	else k_PhotonPass<false><<< n, p0 >>>(spp);
	cudaThreadSynchronize();
	cudaMemcpyFromSymbol(&m_sMaps, g_Map, sizeof(k_PhotonMapCollection));
}