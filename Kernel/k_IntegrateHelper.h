#pragma once

#include "k_TraceHelper.h"

CUDA_ONLY_FUNC bool Occluded(const e_VisibilitySegment& seg)
{
	TraceResult r2;
	r2.Init();
	return k_TraceRay<true>(seg.r.direction, seg.r.origin, &r2) &&  seg.IsValidHit(r2.m_fDist * 1.05f);//seg.tmax > r2.m_fDist && abs(r2.m_fDist - seg.tmax) > 0.1f;
}

CUDA_ONLY_FUNC float3 Transmittance(const Ray& r, float tmin, float tmax)
{
	if(g_SceneData.m_sVolume.HasVolumes())
	{
		float a, b;
		g_SceneData.m_sVolume.IntersectP(r, tmin, tmax, &a, &b);
		return exp(g_SceneData.m_sVolume.tau(r, a, b));
	}
	return make_float3(1);
}

CUDA_ONLY_FUNC float3 Transmittance(const e_VisibilitySegment& seg)
{
	return Transmittance(seg.r, seg.tmin, seg.tmax);
}

CUDA_ONLY_FUNC float3 EstimateDirect(const float3& p, const float3& n, const float3& wo, const e_KernelBSDF* bsdf, CudaRNG& rng, const e_KernelLight* light, unsigned int li, const LightSample& lightSample, const BSDFSample& bsdfSample, BxDFType flags)
{
	float3 Ld = make_float3(0);
	float3 wi;
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
			if(k_TraceRay<true>(wi, p, &r2) && LightIndex(r2, g_SceneData) == li)
				Li = Le(p, n, -wi, r2, g_SceneData);
				//Li = light->L(p, n, wi);
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

CUDA_ONLY_FUNC float3 UniformSampleOneLight(const float3& p, const float3& n, const float3& wo, const e_KernelBSDF* bsdf, CudaRNG& rng)
{
	int nLights = g_SceneData.m_sLightData.UsedCount;
    if (nLights == 0)
		return make_float3(0.0f);
    int lightNum = Floor2Int(rng.randomFloat() * nLights);
    lightNum = MIN(lightNum, nLights-1);
    e_KernelLight *light = g_SceneData.m_sLightData.Data + lightNum;
	LightSample lightSample(rng);
    BSDFSample bsdfSample(rng);
	return float(nLights) * EstimateDirect(p, n, wo, bsdf, rng, light, lightNum, lightSample, bsdfSample, BxDFType(BSDF_ALL & ~BSDF_SPECULAR));
}

CUDA_ONLY_FUNC float3 UniformSampleAllLights(const float3& p, const float3& n, const float3& wo, const e_KernelBSDF* bsdf, CudaRNG& rng, int nSamples)
{
	float3 L = make_float3(0);
	for(int i = 0; i < g_SceneData.m_sLightData.UsedCount; i++)
	{
		e_KernelLight* light = g_SceneData.m_sLightData.Data + i;
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