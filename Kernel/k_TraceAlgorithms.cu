#include "k_TraceAlgorithms.h"

Spectrum EstimateDirect(BSDFSamplingRecord& bRec, const e_KernelMaterial& mat, const e_KernelLight* light, unsigned int li, EBSDFType flags)
{
	DirectSamplingRecord dRec(bRec.map.P, bRec.map.sys.n);
	Spectrum value = light->sampleDirect(dRec, bRec.rng->randomFloat2());
	if(!value.isZero())
	{
		bRec.wo = bRec.map.sys.toLocal(dRec.d);
		bRec.typeMask = flags;
		Spectrum bsdfVal = mat.bsdf.f(bRec);
		if (!bsdfVal.isZero() && !g_SceneData.Occluded(Ray(dRec.ref, dRec.d), 0, dRec.dist))
		{
			const float bsdfPdf = mat.bsdf.pdf(bRec);
			const float weight = MonteCarlo::PowerHeuristic(1, dRec.pdf, 1, bsdfPdf);
			return value * bsdfVal * weight;
		}
	}
	bRec.typeMask = EAll;
	return 0.0f;
	/*
	Spectrum Ld = make_float3(0.0f);
	float lightPdf, bsdfPdf;
	DirectSamplingRecord dRec(bRec.map.P, bRec.map.sys.n, bRec.map.uv);
	Spectrum Li = light->sampleDirect(dRec, bRec.rng->randomFloat2());
	lightPdf = dRec.pdf;
	if(lightPdf > 0.0f && !Li.isZero())
	{
		bRec.wo = bRec.map.sys.toLocal(dRec.d);
		Spectrum f = mat.bsdf.f(bRec);
		Ray r(dRec.ref, dRec.d);
		if(!f.isZero() && !Occluded(r, 0, dRec.dist))
		{
			Li = Li * Transmittance(r, 0, dRec.dist);
			if(light->IsDeltaLight())
				Ld += f * Li * AbsDot(r.direction, bRec.map.sys.n);
			else
			{
				bRec.typeMask = flags;
				bsdfPdf = mat.bsdf.pdf(bRec);
				float weight = MonteCarlo::PowerHeuristic(1, lightPdf, 1, bsdfPdf);
				Ld += f * Li * AbsDot(r.direction, bRec.map.sys.n) * weight;
				bRec.typeMask = EAll;
			}
		}
	}
	
	if(!light->IsDeltaLight())
	{
		bRec.typeMask = flags;
		Spectrum f = mat.bsdf.sample(bRec, bRec.rng->randomFloat2());
		float3 wi = bRec.map.sys.toWorld(bRec.wo);
		if(!f.isZero() && bsdfPdf > 0.0f)
		{
			float weight = 1.0f;
			if (!(bRec.sampledType & EDelta))
			{
                if (lightPdf == 0.0f)
                    return Ld;
                weight = MonteCarlo::PowerHeuristic(1, bsdfPdf, 1, lightPdf);
            }
			Spectrum Li = make_float3(0.0f);
			TraceResult r2;
			r2.Init();
			if(k_TraceRay(wi, bRec.map.P, &r2) && r2.LightIndex() == li)
				Li = r2.Le(bRec.map.P, bRec.map.sys.n, -wi);
			else Li = light->eval(bRec.map.P, bRec.map.sys, wi);
			if(!Li.isZero())
			{
				Li = Li * Transmittance(Ray(bRec.map.P, wi), 0, r2.m_fDist);
				Ld += Li * f * AbsDot(wi, bRec.map.sys.n) * weight;
			}
		}
	}

	return Ld;*/
}

Spectrum UniformSampleAllLights(BSDFSamplingRecord& bRec, const e_KernelMaterial& mat, int nSamples)
{
	//only sample the relevant lights and assume the others emit the same
	Spectrum L = Spectrum(0.0f);
	for(unsigned int i = 0; i < g_SceneData.m_uEmitterCount; i++)
	{
		unsigned int l = g_SceneData.m_uEmitterIndices[i];
		e_KernelLight* light = g_SceneData.m_sLightData.Data + l;
		Spectrum Ld = Spectrum(0.0f);
		for(int j = 0; j < nSamples; j++)
		{
			Ld += EstimateDirect(bRec, mat, light, l, EBSDFType(EAll & ~EDelta));
		}
		L += Ld / float(nSamples);
	}
	return L * float(g_SceneData.m_sLightData.UsedCount) / float(g_SceneData.m_uEmitterCount);
}

Spectrum UniformSampleOneLight(BSDFSamplingRecord& bRec, const e_KernelMaterial& mat)
{
	if(!g_SceneData.m_uEmitterCount)
		return 0.0f;
	float emitpdf;
	unsigned int index = g_SceneData.m_uEmitterIndices[g_SceneData.m_emitterPDF.SampleDiscrete(bRec.rng->randomFloat(), &emitpdf)];
	return float(g_SceneData.m_sLightData.UsedCount) * EstimateDirect(bRec, mat, g_SceneData.m_sLightData.Data + index, index, EBSDFType(EAll & ~EDelta));
}