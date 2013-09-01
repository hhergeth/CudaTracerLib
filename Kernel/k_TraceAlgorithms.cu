#include "k_TraceAlgorithms.h"

float3 EstimateDirect(BSDFSamplingRecord& bRec, const e_KernelMaterial& mat, const e_KernelLight* light, unsigned int li, const LightSample& lightSample, EBSDFType flags)
{
	float3 Ld = make_float3(0);
	float lightPdf, bsdfPdf;
	e_VisibilitySegment seg;
	float3 Li = light->Sample_L(g_SceneData, bRec.map.P, lightSample, &lightPdf, &seg);
	if(lightPdf > 0.0f && !ISBLACK(Li))
	{
		bRec.wo = seg.r.direction;
		float3 f = mat.bsdf.f(bRec);
		if(!ISBLACK(f) && !Occluded(seg))
		{
			Li = Li * Transmittance(seg);
			if(light->IsDeltaLight())
				Ld += f * Li * (AbsDot(seg.r.direction, bRec.map.sys.n) / lightPdf);
			else
			{
				bRec.typeMask = flags;
				bsdfPdf = mat.bsdf.pdf(bRec);
				float weight = PowerHeuristic(1, lightPdf, 1, bsdfPdf);
				Ld += f / bsdfPdf * Li * (AbsDot(seg.r.direction, bRec.map.sys.n) * weight / lightPdf);
				bRec.typeMask = EAll;
			}
		}
	}
	
	float3 wi;
	if(!light->IsDeltaLight())
	{
		float3 f = mat.bsdf.sample(bRec, bRec.rng->randomFloat2());
		if(!ISBLACK(f) && bsdfPdf > 0.0f)
		{
			float weight = 1.0f;
			if (!(bRec.sampledType & EDelta))
			{
                lightPdf = light->Pdf(g_SceneData, bRec.map.P, wi);
                if (lightPdf == 0.0f)
                    return Ld;
                weight = PowerHeuristic(1, bsdfPdf, 1, lightPdf);
            }
			float3 Li = make_float3(0.0f);
			TraceResult r2;
			r2.Init();
			if(k_TraceRay(wi, bRec.map.P, &r2) && r2.LightIndex() == li)
				Li = r2.Le(bRec.map.P, bRec.map.sys.n, -wi);
			else Li = light->Le(g_SceneData, Ray(bRec.map.P, wi));
			if(!ISBLACK(Li))
			{
				Li = Li * Transmittance(Ray(bRec.map.P, wi), 0, r2.m_fDist);
				//not shure about the / bsdfPdf
				Ld += Li * f / bsdfPdf * AbsDot(wi, bRec.map.sys.n) * weight / bsdfPdf;
			}
		}
	}

	return Ld;
}

float3 UniformSampleAllLights(BSDFSamplingRecord& bRec, const e_KernelMaterial& mat, int nSamples)
{
	float3 L = make_float3(0);
	for(int i = 0; i < g_SceneData.m_sLightSelector.m_uCount; i++)
	{
		e_KernelLight* light = g_SceneData.m_sLightData.Data + g_SceneData.m_sLightSelector.m_sIndices[i];
		float3 Ld = make_float3(0);
		for(int j = 0; j < nSamples; j++)
		{
			LightSample lightSample(*bRec.rng);
			Ld += EstimateDirect(bRec, mat, light, i, lightSample, EBSDFType(EAll & ~EDelta));
		}
		L += Ld / float(nSamples);
	}
	return L;
}

float3 UniformSampleOneLight(BSDFSamplingRecord& bRec, const e_KernelMaterial& mat)
{
	int nLights = g_SceneData.m_sLightSelector.m_uCount;
    if (nLights == 0)
		return make_float3(0.0f);
	int lightNum = Floor2Int(bRec.rng->randomFloat() * nLights);
    lightNum = MIN(lightNum, nLights-1);
	e_KernelLight *light = g_SceneData.m_sLightData.Data + g_SceneData.m_sLightSelector.m_sIndices[lightNum];
	LightSample lightSample(*bRec.rng);
	return float(nLights) * EstimateDirect(bRec, mat, light, lightNum, lightSample, EBSDFType(EAll & ~EDelta));
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
	BSDFSamplingRecord bRec;
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
		r.getBsdfSample(r0, rnd, &bRec);
		if(!DIRECT || (depth == 1 || specularBounce))
			cl += cf * r.Le(r0(r.m_fDist), bRec.map.sys.n, -r0.direction);
		if(DIRECT)
			cl += cf * UniformSampleAllLights(bRec, r.getMat(), 1);
		float3 f = r.getMat().bsdf.sample(bRec, rnd.randomFloat2());
		specularBounce = (bRec.sampledType & EDelta) != 0;
		float p = f.x>f.y && f.x>f.z ? f.x : f.y>f.z ? f.y : f.z; 
		if (depth > 5)
			if (rnd.randomFloat() < p)
				f = f / p;
			else break;
		if(ISBLACK(f))
			break;
		cf = cf * f;
		r0 = Ray(r0(r.m_fDist), bRec.wo);
	}
	return cl;
}