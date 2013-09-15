#include "k_TraceAlgorithms.h"

Spectrum EstimateDirect(BSDFSamplingRecord& bRec, const e_KernelMaterial& mat, const e_KernelLight* light, unsigned int li, const LightSample& lightSample, EBSDFType flags)
{
	Spectrum Ld = make_float3(0.0f);
	float lightPdf, bsdfPdf;
	e_VisibilitySegment seg;
	Spectrum Li = light->Sample_L(g_SceneData, bRec.map.P, lightSample, &lightPdf, &seg);
	if(lightPdf > 0.0f && !Li.isZero())
	{
		bRec.wo = bRec.map.sys.toLocal(seg.r.direction);
		Spectrum f = mat.bsdf.f(bRec);
		if(!f.isZero() && !Occluded(seg))
		{
			Li = Li * Transmittance(seg);
			if(light->IsDeltaLight())
				Ld += f * Li * (AbsDot(seg.r.direction, bRec.map.sys.n) / lightPdf);
			else
			{
				bRec.typeMask = flags;
				bsdfPdf = mat.bsdf.pdf(bRec);
				float weight = MonteCarlo::PowerHeuristic(1, lightPdf, 1, bsdfPdf);
				Ld += f / bsdfPdf * Li * (AbsDot(seg.r.direction, bRec.map.sys.n) * weight / lightPdf);
				bRec.typeMask = EAll;
			}
		}
	}
	
	if(!light->IsDeltaLight())
	{
		Spectrum f = mat.bsdf.sample(bRec, bRec.rng->randomFloat2());
		float3 wi = bRec.map.sys.toWorld(bRec.wo);
		if(!f.isZero() && bsdfPdf > 0.0f)
		{
			float weight = 1.0f;
			if (!(bRec.sampledType & EDelta))
			{
                lightPdf = light->Pdf(g_SceneData, bRec.map.P, wi);
                if (lightPdf == 0.0f)
                    return Ld;
                weight = MonteCarlo::PowerHeuristic(1, bsdfPdf, 1, lightPdf);
            }
			Spectrum Li = make_float3(0.0f);
			TraceResult r2;
			r2.Init();
			if(k_TraceRay(wi, bRec.map.P, &r2) && r2.LightIndex() == li)
				Li = r2.Le(bRec.map.P, bRec.map.sys.n, -wi);
			else Li = light->Le(g_SceneData, Ray(bRec.map.P, wi));
			if(!Li.isZero())
			{
				Li = Li * Transmittance(Ray(bRec.map.P, wi), 0, r2.m_fDist);
				//not shure about the / bsdfPdf
				Ld += Li * f / bsdfPdf * AbsDot(wi, bRec.map.sys.n) * weight / bsdfPdf;
			}
		}
	}

	return Ld;
}

Spectrum UniformSampleAllLights(BSDFSamplingRecord& bRec, const e_KernelMaterial& mat, int nSamples)
{
	Spectrum L = Spectrum(0.0f);
	for(unsigned int i = 0; i < g_SceneData.m_sLightSelector.m_uCount; i++)
	{
		e_KernelLight* light = g_SceneData.m_sLightData.Data + g_SceneData.m_sLightSelector.m_sIndices[i];
		Spectrum Ld = Spectrum(0.0f);
		for(int j = 0; j < nSamples; j++)
		{
			LightSample lightSample(*bRec.rng);
			Ld += EstimateDirect(bRec, mat, light, i, lightSample, EBSDFType(EAll & ~EDelta));
		}
		L += Ld / float(nSamples);
	}
	return L;
}

Spectrum UniformSampleOneLight(BSDFSamplingRecord& bRec, const e_KernelMaterial& mat)
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

Spectrum PathTrace(float3& a_Dir, float3& a_Ori, CudaRNG& rnd, float* distTravalled)
{
	const bool DIRECT = false;
	Ray r0 = Ray(a_Ori, a_Dir);
	TraceResult r;
	r.Init(true);
	Spectrum cl = Spectrum(0.0f);   // accumulated color
	Spectrum cf = Spectrum(1.0f);  // accumulated reflectance
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
		Spectrum f = r.getMat().bsdf.sample(bRec, rnd.randomFloat2());
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
	}
	return cl;
}