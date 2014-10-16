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

CUDA_HOST CUDA_DEVICE Spectrum UniformSampleAllLights(const BSDFSamplingRecord& bRec, const e_KernelMaterial& mat, int nSamples);

CUDA_HOST CUDA_DEVICE Spectrum UniformSampleOneLight(const BSDFSamplingRecord& bRec, const e_KernelMaterial& mat);

template<bool DIRECT> CUDA_FUNC_IN Spectrum PathTrace(float3& a_Dir, float3& a_Ori, CudaRNG& rnd, float* distTravalled = 0)
{
	Ray r0 = Ray(a_Ori, a_Dir);
	TraceResult r;
	r.Init();
	Spectrum cl = Spectrum(0.0f);   // accumulated color
	Spectrum cf = Spectrum(1.0f);  // accumulated reflectance
	int depth = 0;
	bool specularBounce = false;
	BSDFSamplingRecord bRec;
	float accPdf = 1;
	while (k_TraceRay(r0.direction, r0.origin, &r) && depth++ < 7)
	{
		if(distTravalled && depth == 1)
			*distTravalled = r.m_fDist;
		r.getBsdfSample(r0, rnd, &bRec); //return (Spectrum(bRec.map.sys.n) + Spectrum(1)) / 2.0f; //return bRec.map.sys.n;
		if(!DIRECT || (depth == 1 || specularBounce))
			cl += cf * r.Le(r0(r.m_fDist), bRec.map.sys, -r0.direction);
		Spectrum f = r.getMat().bsdf.sample(bRec, rnd.randomFloat2());
		if(DIRECT)
			cl += cf * UniformSampleAllLights(bRec, r.getMat(), 1);
		accPdf *= r.getMat().bsdf.pdf(bRec);
		specularBounce = (bRec.sampledType & EDelta) != 0;
		float p = f.max(); 
		if (depth > 5)
			if (rnd.randomFloat() < p)
				f = f / p;
			else break;
		if(f.isZero())
			break;
		cf = cf * f;
		r0 = Ray(r0(r.m_fDist), bRec.getOutgoing());
		r.Init();
	}
	if(!r.hasHit())
		cl += cf * g_SceneData.EvalEnvironment(r0);
	//return accPdf;
	return cl;
}