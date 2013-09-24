#pragma once

#include "k_TraceHelper.h"

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

CUDA_HOST CUDA_DEVICE Spectrum EstimateDirect(BSDFSamplingRecord& bRec, const e_KernelMaterial& mat, const e_KernelLight* light, unsigned int li, EBSDFType flags);

CUDA_HOST CUDA_DEVICE Spectrum UniformSampleAllLights(BSDFSamplingRecord& bRec, const e_KernelMaterial& mat, int nSamples);

CUDA_HOST CUDA_DEVICE Spectrum UniformSampleOneLight(BSDFSamplingRecord& bRec, const e_KernelMaterial& mat);

CUDA_HOST CUDA_DEVICE Spectrum PathTrace(float3& a_Dir, float3& a_Ori, CudaRNG& rnd, float* distTravalled = 0);