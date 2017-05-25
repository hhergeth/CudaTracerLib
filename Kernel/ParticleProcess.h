#pragma once

#include <Kernel/TraceHelper.h>
#include <Base/CudaRandom.h>
#include <SceneTypes/Light.h>

namespace CudaTracerLib {

/*

//Example particle process handler
struct ParticleProcessHandler
{
	CUDA_FUNC_IN void handleEmission(const Spectrum& weight, const PositionSamplingRecord& pRec)
	{

	}

	CUDA_FUNC_IN void handleSurfaceInteraction(const Spectrum& weight, const Spectrum& f, const TraceResult& res, BSDFSamplingRecord& bRec, const TraceResult& r2, bool lastBssrdf, bool lastDelta)
	{

	}

	CUDA_FUNC_IN void handleMediumSampling(const Spectrum& weight, const NormalizedT<Ray>& r, const TraceResult& r2, const MediumSamplingRecord& mRec, bool sampleInMedium, const VolumeRegion* bssrdf, bool lastDelta)
	{

	}

	CUDA_FUNC_IN void handleMediumInteraction(const Spectrum& weight, const Spectrum& f, const MediumSamplingRecord& mRec, const NormalizedT<Vec3f>& wi, const TraceResult& r2, const VolumeRegion* bssrdf, bool lastDelta)
	{

	}
};

*/

template<bool NEEDS_EMISSION_SAMPLE, bool PARTICIPATING_MEDIA = true, bool SUBSURFACE_SCATTERING = true, typename PROCESS> CUDA_FUNC_IN void ParticleProcess(int maxDepth, int rrStartDepth, Sampler& rng, PROCESS& P)
{
	NormalizedT<Ray> r;

	Spectrum power, throughput = Spectrum(1.0f);
	if(NEEDS_EMISSION_SAMPLE)
	{
		PositionSamplingRecord pRec;
		power = g_SceneData.sampleEmitterPosition(pRec, rng.randomFloat2());

		P.handleEmission(power, pRec);

		DirectionSamplingRecord dRec;
		power *= ((const Light*)pRec.object)->sampleDirection(dRec, pRec, rng.randomFloat2());

		r = NormalizedT<Ray>(pRec.p, dRec.d);
	}
	else
	{
		power = g_SceneData.sampleEmitterRay(r, rng.randomFloat2(), rng.randomFloat2());
	}

	int depth = -1;
	BSDFSamplingRecord bRec;

	KernelAggregateVolume& V = g_SceneData.m_sVolume;
	MediumSamplingRecord mRec;
	const VolumeRegion* bssrdf = 0;
	bool delta = false;

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
			P.handleMediumSampling(power * throughput, r, r2, mRec, distInMedium, bssrdf, delta);

		if (distInMedium)
		{
			throughput *= mRec.sigmaS * mRec.transmittance / mRec.pdfSuccess;
			PhaseFunctionSamplingRecord pfRec(-r.dir());
			float pdf;
			Spectrum f;
			if (bssrdf)
				f = bssrdf->As()->Func.Sample(pfRec, pdf, rng.randomFloat2());
			else f = V.Sample(mRec.p, pfRec, pdf, rng.randomFloat2());
			P.handleMediumInteraction(power * throughput, f, mRec, -r.dir(), r2, bssrdf, delta);
			throughput *= f;
			r.dir() = pfRec.wo;
			r.ori() = mRec.p;
			delta = false;
		}
		else if (!r2.hasHit())
			break;
		else
		{
			if (sampledDistance)
			{
				throughput *= mRec.transmittance / mRec.pdfFailure;
			}
			auto wo = bssrdf ? -r.dir() : r.dir();
			Spectrum f_i = power * throughput;
			r2.getBsdfSample(wo, r(r2.m_fDist), bRec, ETransportMode::EImportance, &f_i);
			Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());//do it before calling to handler to make the sampling type available to the handler
			auto woSave = bRec.wo;
			P.handleSurfaceInteraction(power * throughput, f, r, r2, bRec, !!bssrdf, delta);
			bRec.wo = woSave;
			if (SUBSURFACE_SCATTERING && !bssrdf && r2.getMat().GetBSSRDF(bRec.dg, &bssrdf))
				bRec.wo.z *= -1.0f;
			else
			{
				if (!bssrdf)
				{
					throughput *= f;
				}
				bssrdf = 0;
			}
			if (throughput.isZero())
				break;
			delta = (bRec.sampledType & ETypeCombinations::EDelta) != 0;

			if (depth >= rrStartDepth)
			{
				float q = min(throughput.max(), 0.95f);
				if (rng.randomFloat() >= q)
					break;
				throughput /= q;
			}

			r = NormalizedT<Ray>(bRec.dg.P, bRec.getOutgoing());
		}
	}
}

}