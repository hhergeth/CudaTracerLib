#pragma once

#include "k_TraceHelper.h"

CUDA_FUNC_IN bool V(const float3& a, const float3& b)
{
	float3 d = b - a;
	float l = length(d);
	return !g_SceneData.Occluded(Ray(a, d / l), 0, l);
}

CUDA_FUNC_IN float G(const float3& N_x, const float3& N_y, const float3& x, const float3& y)
{
	float3 theta = normalize(y - x);
	return AbsDot(N_x, theta) * AbsDot(N_y, -theta) / DistanceSquared(x, y);
}

CUDA_FUNC_IN Spectrum Transmittance(const Ray& r, float tmin, float tmax)
{
	if(g_SceneData.m_sVolume.HasVolumes())
	{
		float a, b;
		g_SceneData.m_sVolume.IntersectP(r, tmin, tmax, &a, &b);
		return g_SceneData.m_sVolume.tau(r, a, b).exp();
	}
	return make_float3(1);
}

CUDA_HOST CUDA_DEVICE Spectrum UniformSampleAllLights(const BSDFSamplingRecord& bRec, const e_KernelMaterial& mat, int nSamples);

CUDA_HOST CUDA_DEVICE Spectrum UniformSampleOneLight(const BSDFSamplingRecord& bRec, const e_KernelMaterial& mat);