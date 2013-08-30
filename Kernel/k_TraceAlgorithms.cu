#include "k_TraceAlgorithms.h"

float3 EstimateDirect(const float3& p, const float3& n, const float3& wo, const e_KernelBSDF* bsdf, CudaRNG& rng, const e_KernelLight* light, unsigned int li, const LightSample& lightSample, const BSDFSample& bsdfSample, BxDFType flags)
{
	float3 Ld = make_float3(0);
	float lightPdf, bsdfPdf;
	e_VisibilitySegment seg;
	float3 Li = light->Sample_L(g_SceneData, p, lightSample, &lightPdf, &seg);
	if(lightPdf > 0.0f && !ISBLACK(Li))
	{
		float3 f = bsdf->f(wo, seg.r.direction, flags);
		if(!ISBLACK(f) && !Occluded(seg))
		{
			Li = Li * Transmittance(seg);
			if(light->IsDeltaLight())
				Ld += f * Li * (AbsDot(seg.r.direction, n) / lightPdf);
			else
			{
				bsdfPdf = bsdf->Pdf(wo, seg.r.direction, flags);
				float weight = PowerHeuristic(1, lightPdf, 1, bsdfPdf);
				Ld += f * Li * (AbsDot(seg.r.direction, n) * weight / lightPdf);
			}
		}
	}
	
	float3 wi;
	if(!light->IsDeltaLight())
	{
		BxDFType sampledType;
        float3 f = bsdf->Sample_f(wo, &wi, bsdfSample, &bsdfPdf, flags, &sampledType);
		if(!ISBLACK(f) && bsdfPdf > 0.0f)
		{
			float weight = 1.0f;
			if (!(sampledType & BSDF_SPECULAR))
			{
                lightPdf = light->Pdf(g_SceneData, p, wi);
                if (lightPdf == 0.0f)
                    return Ld;
                weight = PowerHeuristic(1, bsdfPdf, 1, lightPdf);
            }
			float3 Li = make_float3(0.0f);
			TraceResult r2;
			r2.Init();
			if(k_TraceRay(wi, p, &r2) && r2.LightIndex() == li)
				Li = r2.Le(p, n, -wi);
			else Li = light->Le(g_SceneData, Ray(p, wi));
			if(!ISBLACK(Li))
			{
				Li = Li * Transmittance(Ray(p, wi), 0, r2.m_fDist);
				Ld += Li * f * AbsDot(wi, n) * weight / bsdfPdf;
			}
		}
	}

	return Ld;
}

float3 UniformSampleAllLights(const float3& p, const float3& n, const float3& wo, const e_KernelBSDF* bsdf, CudaRNG& rng, int nSamples)
{
	float3 L = make_float3(0);
	for(int i = 0; i < g_SceneData.m_sLightSelector.m_uCount; i++)
	{
		e_KernelLight* light = g_SceneData.m_sLightData.Data + g_SceneData.m_sLightSelector.m_sIndices[i];
		float3 Ld = make_float3(0);
		for(int j = 0; j < nSamples; j++)
		{
			LightSample lightSample(rng);
			BSDFSample bsdfSample(rng);
			Ld += EstimateDirect(p, n, wo, bsdf, rng, light, i, lightSample, bsdfSample, BxDFType(BSDF_ALL & ~BSDF_SPECULAR));
		}
		L += Ld / float(nSamples);
	}
	return L;
}

float3 UniformSampleOneLight(const float3& p, const float3& n, const float3& wo, const e_KernelBSDF* bsdf, CudaRNG& rng)
{
	int nLights = g_SceneData.m_sLightSelector.m_uCount;
    if (nLights == 0)
		return make_float3(0.0f);
    int lightNum = Floor2Int(rng.randomFloat() * nLights);
    lightNum = MIN(lightNum, nLights-1);
	e_KernelLight *light = g_SceneData.m_sLightData.Data + g_SceneData.m_sLightSelector.m_sIndices[lightNum];
	LightSample lightSample(rng);
    BSDFSample bsdfSample(rng);
	return float(nLights) * EstimateDirect(p, n, wo, bsdf, rng, light, lightNum, lightSample, bsdfSample, BxDFType(BSDF_ALL & ~BSDF_SPECULAR));
}

float3 PathTrace(float3& a_Dir, float3& a_Ori, CudaRNG& rnd, float* distTravalled)
{
	const bool DIRECT = true;
	Ray r0 = Ray(a_Ori, a_Dir);
	TraceResult r;
	float3 cl = make_float3(0,0,0);   // accumulated color
	float3 cf = make_float3(1,1,1);  // accumulated reflectance
	int depth = 0;
	bool specularBounce = false;
	while (depth++ < 7)
	{
		r.Init();
		if(!k_TraceRay(r0.direction, r0.origin, &r))
		{
			if(g_SceneData.m_sEnvMap.CanSample())
				cl += cf * g_SceneData.m_sEnvMap.Sample(r0);
			break;
		}
		if(distTravalled && depth == 1)
			*distTravalled = r.m_fDist;
		float3 wi;
		float pdf;
		e_KernelBSDF bsdf = r.GetBSDF(r0(r.m_fDist));
		if(!DIRECT || (depth == 1 || specularBounce))
			cl += cf * r.Le(r0(r.m_fDist), bsdf.ng, -r0.direction);
		if(DIRECT)
			cl += cf * UniformSampleAllLights(r0(r.m_fDist), bsdf.sys.n, -r0.direction, &bsdf, rnd, 1);
		BxDFType flags;
		float3 f = bsdf.Sample_f(-r0.direction, &wi, BSDFSample(rnd), &pdf, BSDF_ALL, &flags);
		specularBounce = (flags & BSDF_SPECULAR) != 0;
		float p = f.x>f.y && f.x>f.z ? f.x : f.y>f.z ? f.y : f.z; 
		if (depth > 5)
			if (rnd.randomFloat() < p)
				f = f / p;
			else break;
		if(!pdf)
			break;
		cf = cf * f * AbsDot(wi, bsdf.sys.n) / pdf;
		r0 = Ray(r0(r.m_fDist), wi);
	}
	return cl;
}