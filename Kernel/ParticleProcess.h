#pragma once

#include <Kernel/TraceHelper.h>
#include <Base/CudaRandom.h>
#include <Engine/Light.h>

namespace CudaTracerLib {

/*

//Example particle process handler
struct ParticleProcessHandler
{
	CUDA_FUNC_IN void handleEmission(const Spectrum& weight, const PositionSamplingRecord& pRec)
	{

	}

	CUDA_FUNC_IN void handleSurfaceInteraction(const Spectrum& weight, float accum_pdf, const Spectrum& f, float pdf, const TraceResult& res, BSDFSamplingRecord& bRec, const TraceResult& r2, bool lastBssrdf)
	{

	}

	CUDA_FUNC_IN void handleMediumSampling(const Spectrum& weight, float accum_pdf, const NormalizedT<Ray>& r, const TraceResult& r2, const MediumSamplingRecord& mRec, bool sampleInMedium, const VolumeRegion* bssrdf)
	{

	}

	CUDA_FUNC_IN void handleMediumInteraction(const Spectrum& weight, float accum_pdf, const Spectrum& f, float pdf, const MediumSamplingRecord& mRec, const NormalizedT<Vec3f>& wi, const TraceResult& r2, const VolumeRegion* bssrdf)
	{

	}
};

*/

template<bool PARTICIPATING_MEDIA = true, bool SUBSURFACE_SCATTERING = true, typename PROCESS> CUDA_FUNC_IN void ParticleProcess(int maxDepth, int rrStartDepth, Sampler& rng, PROCESS& P)
{
	PositionSamplingRecord pRec;
	Spectrum power = g_SceneData.sampleEmitterPosition(pRec, rng.randomFloat2()), throughput = Spectrum(1.0f);

	P.handleEmission(power, pRec);

	DirectionSamplingRecord dRec;
	power *= ((const Light*)pRec.object)->sampleDirection(dRec, pRec, rng.randomFloat2());

	NormalizedT<Ray> r(pRec.p, dRec.d);
	int depth = -1;
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);

	KernelAggregateVolume& V = g_SceneData.m_sVolume;
	MediumSamplingRecord mRec;
	const VolumeRegion* bssrdf = 0;
	float accum_pdf = pRec.pdf * dRec.pdf;//the pdf of the complete path up to the current vertex

	while (++depth < maxDepth && !throughput.isZero())
	{
		TraceResult r2 = traceRay(r);
		float minT, maxT;
		bool distInMedium = false, sampledDistance = false;
		if (PARTICIPATING_MEDIA && !bssrdf && V.HasVolumes() && V.IntersectP(r, 0, r2.m_fDist, &minT, &maxT))
		{
			sampledDistance = true;
			distInMedium = V.sampleDistance(r, 0, r2.m_fDist, rng.randomFloat(), mRec);
		}
		else if (bssrdf)
		{
			sampledDistance = true;
			distInMedium = bssrdf->sampleDistance(r, 0, r2.m_fDist, rng.randomFloat(), mRec);
		}

		if (sampledDistance)
		{
			P.handleMediumSampling(power * throughput, accum_pdf, r, r2, mRec, distInMedium, bssrdf);
		}

		if (distInMedium)
		{
			throughput *= mRec.sigmaS * mRec.transmittance / mRec.pdfSuccess;
			accum_pdf *= mRec.pdfSuccess;
			PhaseFunctionSamplingRecord pfRec(-r.dir());
			float pdf;
			Spectrum f;
			if (bssrdf)
				f = bssrdf->As()->Func.Sample(pfRec, pdf, rng.randomFloat2());
			else f = V.Sample(mRec.p, pfRec, pdf, rng.randomFloat2());
			P.handleMediumInteraction(power * throughput, accum_pdf, f, pdf, mRec, -r.dir(), r2, bssrdf);
			accum_pdf *= pdf;
			throughput *= f;
			r.dir() = pfRec.wo;
			r.ori() = mRec.p;
		}
		else if (!r2.hasHit())
			break;
		else
		{
			if (sampledDistance)
			{
				throughput *= mRec.transmittance / mRec.pdfFailure;
				accum_pdf *= mRec.pdfFailure;
			}
			auto wo = bssrdf ? -r.dir() : r.dir();
			Spectrum f_i = power * throughput;
			r2.getBsdfSample(wo, r(r2.m_fDist), bRec, ETransportMode::EImportance, &f_i);
			float pdf;
			Spectrum f = r2.getMat().bsdf.sample(bRec, pdf, rng.randomFloat2());//do it before calling to handler to make the sampling type available to the handler
			auto woSave = bRec.wo;
			P.handleSurfaceInteraction(power * throughput, accum_pdf, f, pdf, r, r2, bRec, !!bssrdf);
			bRec.wo = woSave;
			if (SUBSURFACE_SCATTERING && !bssrdf && r2.getMat().GetBSSRDF(bRec.dg, &bssrdf))
				bRec.wo.z *= -1.0f;
			else
			{
				if (!bssrdf)
				{
					throughput *= f;
					accum_pdf *= pdf;
				}
				bssrdf = 0;
			}
			if (throughput.isZero())
				break;

			if (depth >= rrStartDepth)
			{
				float q = min(throughput.max(), 0.95f);
				if (rng.randomFloat() >= q)
					break;
				throughput /= q;
				accum_pdf *= q;
			}

			r = NormalizedT<Ray>(bRec.dg.P, bRec.getOutgoing());
		}
	}
}

}