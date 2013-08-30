#pragma once

#include "k_TraceHelper.h"

CUDA_FUNC_IN bool Occluded(const e_VisibilitySegment& seg)
{
	TraceResult r2;
	r2.Init();
	return k_TraceRay(seg.r.direction, seg.r.origin, &r2) &&  seg.IsValidHit(r2.m_fDist * 1.05f);//seg.tmax > r2.m_fDist && abs(r2.m_fDist - seg.tmax) > 0.1f;
}

CUDA_FUNC_IN float3 Transmittance(const Ray& r, float tmin, float tmax)
{
	if(g_SceneData.m_sVolume.HasVolumes())
	{
		float a, b;
		g_SceneData.m_sVolume.IntersectP(r, tmin, tmax, &a, &b);
		return exp(g_SceneData.m_sVolume.tau(r, a, b));
	}
	return make_float3(1);
}

CUDA_FUNC_IN float3 Transmittance(const e_VisibilitySegment& seg)
{
	return Transmittance(seg.r, seg.tmin, seg.tmax);
}

CUDA_HOST CUDA_DEVICE float3 EstimateDirect(const float3& p, const float3& n, const float3& wo, const e_KernelBSDF* bsdf, CudaRNG& rng, const e_KernelLight* light, unsigned int li, const LightSample& lightSample, const BSDFSample& bsdfSample, BxDFType flags);

CUDA_HOST CUDA_DEVICE float3 UniformSampleAllLights(const float3& p, const float3& n, const float3& wo, const e_KernelBSDF* bsdf, CudaRNG& rng, int nSamples);

CUDA_HOST CUDA_DEVICE float3 UniformSampleOneLight(const float3& p, const float3& n, const float3& wo, const e_KernelBSDF* bsdf, CudaRNG& rng);

CUDA_HOST CUDA_DEVICE float3 PathTrace(float3& a_Dir, float3& a_Ori, CudaRNG& rnd, float* distTravalled = 0);