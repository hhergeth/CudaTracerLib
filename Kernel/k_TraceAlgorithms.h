#pragma once

#include "k_TraceHelper.h"

CUDA_FUNC_IN bool V(const Vec3f& a, const Vec3f& b, TraceResult* res = 0)
{
	Vec3f d = b - a;
	float l = length(d);
	return !g_SceneData.Occluded(Ray(a, d / l), 0, l, res);
}

CUDA_FUNC_IN float G(const Vec3f& N_x, const Vec3f& N_y, const Vec3f& x, const Vec3f& y)
{
	Vec3f theta = normalize(y - x);
	return absdot(N_x, theta) * absdot(N_y, -theta) / distanceSquared(x, y);
}

CUDA_FUNC_IN Spectrum Transmittance(const Ray& r, float tmin, float tmax, unsigned int a_NodeIndex = 0xffffffff)
{
	if(g_SceneData.m_sVolume.HasVolumes())
	{
		float a, b;
		g_SceneData.m_sVolume.IntersectP(r, tmin, tmax, &a, &b, a_NodeIndex);
		return (-g_SceneData.m_sVolume.tau(r, a, b, a_NodeIndex)).exp();
	}
	return Spectrum(1.0f);
}

CUDA_HOST CUDA_DEVICE Spectrum UniformSampleAllLights(const BSDFSamplingRecord& bRec, const e_KernelMaterial& mat, int nSamples);

CUDA_HOST CUDA_DEVICE Spectrum UniformSampleOneLight(const BSDFSamplingRecord& bRec, const e_KernelMaterial& mat);