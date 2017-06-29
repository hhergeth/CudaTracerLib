#include "TraceAlgorithms.h"
#include "TraceHelper.h"
#include <SceneTypes/Samples.h>
#include <Engine/Material.h>
#include <SceneTypes/Light.h>

namespace CudaTracerLib {

bool V(const Vec3f& a, const Vec3f& b, TraceResult* res)
{
	Vec3f d = b - a;
	float l = length(d);
	return !g_SceneData.Occluded(Ray(a, d / l), 0, l, res);
}

float G(const NormalizedT<Vec3f>& N_x, const NormalizedT<Vec3f>& N_y, const Vec3f& x, const Vec3f& y)
{
	auto theta = normalize(y - x);
	return absdot(N_x, theta) * absdot(N_y, -theta) / distanceSquared(x, y);
}

Spectrum Transmittance(const Ray& r, float tmin, float tmax)
{
	if (g_SceneData.m_sVolume.HasVolumes())
	{
		float a, b;
		g_SceneData.m_sVolume.IntersectP(r, tmin, tmax, &a, &b);
		return (-g_SceneData.m_sVolume.tau(r, a, b)).exp();
	}
	return Spectrum(1.0f);
}

DirectSamplingRecord DirectSamplingRecFromRay(const NormalizedT<Ray>& r, float dist, const NormalizedT<Vec3f>& last_nor, const Vec3f& P, const NormalizedT<Vec3f>& n, EMeasure measure)
{
	DirectSamplingRecord dRec(r.ori(), last_nor);
	dRec.p = P;
	dRec.n = n;
	dRec.d = r.dir();
	dRec.dist = dist;
	dRec.measure = measure;
	return dRec;
}

CUDA_FUNC_IN Spectrum EstimateDirect(BSDFSamplingRecord bRec, const Material& mat, const Light* light, float light_pdf, EBSDFType flags, Sampler& rng, bool attenuated, bool use_mis)
{
	DirectSamplingRecord dRec(bRec.dg.P, bRec.dg.sys.n);
	Spectrum value = light->sampleDirect(dRec, rng.randomFloat2());
	Spectrum retVal(0.0f);
	if (!value.isZero())
	{
		auto oldWo = bRec.wo;
		bRec.wo = bRec.dg.toLocal(dRec.d);
		bRec.typeMask = flags;
		Spectrum bsdfVal = mat.bsdf.f(bRec);
		if (!bsdfVal.isZero() && !g_SceneData.Occluded(Ray(dRec.ref, dRec.d), 0, dRec.dist))
		{
			float weight = 1.0f;
			if (use_mis && dRec.measure != EDiscrete)//compute MIS weight
			{
				const float bsdfPdf = mat.bsdf.pdf(bRec);
				const float directPdf = (dRec.measure == EArea ? PdfAtoW(dRec.pdf, dRec.dist, dot(dRec.n, dRec.d)) : dRec.pdf) * light_pdf;
				weight = MonteCarlo::PowerHeuristic(1, directPdf, 1, bsdfPdf);
			}

			retVal = value * bsdfVal * weight;
			if (attenuated)
				retVal *= Transmittance(Ray(dRec.ref, dRec.d), 0, dRec.dist);
		}
		bRec.typeMask = EAll;
		bRec.wo = oldWo;
	}
	return retVal;
}

Spectrum UniformSampleAllLights(const BSDFSamplingRecord& bRec, const Material& mat, int nSamples, Sampler& rng, bool attenuated, bool use_mis)
{
	//only sample the relevant lights and assume the others emit the same
	Spectrum L = Spectrum(0.0f);
	for (unsigned int i = 0; i < g_SceneData.m_numLights; i++)
	{
		const Light* light = g_SceneData.getLight(i);
		Spectrum Ld = Spectrum(0.0f);
		for (int j = 0; j < nSamples; j++)
		{
			Ld += EstimateDirect((BSDFSamplingRecord&)bRec, mat, light, 1.0f, EBSDFType(EAll & ~EDelta), rng, attenuated, use_mis);
		}
		L += Ld / float(nSamples);
	}
	return L;
}

Spectrum UniformSampleOneLight(const BSDFSamplingRecord& bRec, const Material& mat, Sampler& rng, bool attenuated, bool use_mis)
{
	if (!g_SceneData.m_numLights)
		return Spectrum(0.0f);
	Vec2f sample = rng.randomFloat2();
	float pdf;
	const Light* light = g_SceneData.sampleEmitter(pdf, sample);
	if (light == 0) return Spectrum(0.0f);
	return EstimateDirect((BSDFSamplingRecord&)bRec, mat, light, pdf, EBSDFType(EAll & ~EDelta), rng, attenuated, use_mis) / pdf;
}

}