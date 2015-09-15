#include "k_TraceAlgorithms.h"
#include "k_TraceHelper.h"
#include "../Engine/e_Samples.h"
#include "../Engine/e_Material.h"
#include "../Engine/e_Light.h"

bool V(const Vec3f& a, const Vec3f& b, TraceResult* res)
{
	Vec3f d = b - a;
	float l = length(d);
	return !g_SceneData.Occluded(Ray(a, d / l), 0, l, res);
}

float G(const Vec3f& N_x, const Vec3f& N_y, const Vec3f& x, const Vec3f& y)
{
	Vec3f theta = normalize(y - x);
	return absdot(N_x, theta) * absdot(N_y, -theta) / distanceSquared(x, y);
}

Spectrum Transmittance(const Ray& r, float tmin, float tmax, unsigned int a_NodeIndex)
{
	if (g_SceneData.m_sVolume.HasVolumes())
	{
		float a, b;
		g_SceneData.m_sVolume.IntersectP(r, tmin, tmax, &a, &b, a_NodeIndex);
		return (-g_SceneData.m_sVolume.tau(r, a, b, a_NodeIndex)).exp();
	}
	return Spectrum(1.0f);
}

CUDA_FUNC_IN Spectrum EstimateDirect(BSDFSamplingRecord bRec, const e_KernelMaterial& mat, const e_KernelLight* light, unsigned int li, EBSDFType flags, CudaRNG& rng, bool attenuated)
{
	DirectSamplingRecord dRec(bRec.dg.P, bRec.dg.sys.n);
	Spectrum value = light->sampleDirect(dRec, rng.randomFloat2());
	Spectrum retVal(0.0f);
	if(!value.isZero())
	{
		Vec3f oldWo = bRec.wo;
		bRec.wo = normalize(bRec.dg.toLocal(dRec.d));
		bRec.typeMask = flags;
		Spectrum bsdfVal = mat.bsdf.f(bRec);
		if (!bsdfVal.isZero() && !g_SceneData.Occluded(Ray(dRec.ref, dRec.d), 0, dRec.dist))
		{
			const float bsdfPdf = mat.bsdf.pdf(bRec);
			const float weight = MonteCarlo::PowerHeuristic(1, dRec.pdf, 1, bsdfPdf);
			retVal = value * bsdfVal * weight;
			if (attenuated)
				retVal *= Transmittance(Ray(dRec.ref, dRec.d), 0, dRec.dist);
		}
		bRec.typeMask = EAll;
		bRec.wo = oldWo;
	}
	return retVal;
}

Spectrum UniformSampleAllLights(const BSDFSamplingRecord& bRec, const e_KernelMaterial& mat, int nSamples, CudaRNG& rng, bool attenuated)
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
			Ld += EstimateDirect((BSDFSamplingRecord&)bRec, mat, light, l, EBSDFType(EAll & ~EDelta), rng, attenuated);
		}
		L += Ld / float(nSamples);
	}
	return L;
}

Spectrum UniformSampleOneLight(const BSDFSamplingRecord& bRec, const e_KernelMaterial& mat, CudaRNG& rng, bool attenuated)
{
	if(!g_SceneData.m_uEmitterCount)
		return Spectrum(0.0f);
	float emitpdf;
	unsigned int index = g_SceneData.m_uEmitterIndices[g_SceneData.m_emitterPDF.SampleDiscrete(rng.randomFloat(), &emitpdf)];
	return float(g_SceneData.m_uEmitterCount) * EstimateDirect((BSDFSamplingRecord&)bRec, mat, g_SceneData.m_sLightData.Data + index, index, EBSDFType(EAll & ~EDelta), rng, attenuated);
}