#pragma once

#include "k_TraceHelper.h"

template<bool DIRECT> CUDA_ONLY_FUNC float3 PathTrace(float3& a_Dir, float3& a_Ori, CudaRNG& rnd, float* distTravalled = 0)
{
	Ray r0 = Ray(a_Ori, a_Dir);
	TraceResult r;
	float3 cl = make_float3(0,0,0);   // accumulated color
	float3 cf = make_float3(1,1,1);  // accumulated reflectance
	int depth = 0;
	bool specularBounce = false;
	while (depth++ < 7)
	{
		r.Init();
		if(!k_TraceRay<true>(r0.direction, r0.origin, &r))
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
			cl += cf * UniformSampleAllLights(r0(r.m_fDist), bsdf.sys.m_normal, -r0.direction, &bsdf, rnd, 1);
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
		cf = cf * f * AbsDot(wi, bsdf.sys.m_normal) / pdf;
		r0 = Ray(r0(r.m_fDist), wi);
	}
	return cl;
}