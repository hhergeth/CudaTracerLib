#include "PhotonTracer.h"
#include <Kernel/TraceHelper.h>
#include <Kernel/TraceAlgorithms.h>
#include <Engine/Light.h>
#include <Engine/Sensor.h>

namespace CudaTracerLib {

enum
{
	MaxBlockHeight = 6,
};

CUDA_DEVICE unsigned int g_NextRayCounter3;

CUDA_FUNC_IN void handleEmission(const Spectrum& weight, const PositionSamplingRecord& pRec, Image& g_Image, CudaRNG& rng)
{
	DirectSamplingRecord dRec(pRec.p, pRec.n);
	Spectrum value = weight * g_SceneData.sampleAttenuatedSensorDirect(dRec, rng.randomFloat2());
	if (!value.isZero() && V(dRec.p, dRec.ref))
	{
		const Light* emitter = (const Light*)pRec.object;
		value *= emitter->evalDirection(DirectionSamplingRecord(dRec.d), pRec);
		g_Image.Splat(dRec.uv.x, dRec.uv.y, value);
	}
}

template<bool CORRECT_DIFFERENTIALS> CUDA_FUNC_IN void handleSurfaceInteraction(const Spectrum& weight, const TraceResult& res, BSDFSamplingRecord& bRec, const TraceResult& r2, Image& g_Image, CudaRNG& rng)
{
	DirectSamplingRecord dRec(bRec.dg.P, bRec.dg.sys.n);
	Spectrum value = weight * g_SceneData.sampleAttenuatedSensorDirect(dRec, rng.randomFloat2());
	if (!value.isZero() && V(dRec.p, dRec.ref))
	{
		bRec.wo = bRec.dg.toLocal(dRec.d);

		//compute pixel differentials
		if (CORRECT_DIFFERENTIALS)
		{
			NormalizedT<Ray> r, rX, rY;
			g_SceneData.sampleSensorRay(r, rX, rY, dRec.uv, Vec2f(0));
			auto oldWi = bRec.wi;
			r2.getBsdfSample(r, bRec, ETransportMode::ERadiance, &rng, 0, &dRec.d);
			bRec.dg.computePartials(r, rX, rY);
			bRec.wi = oldWi;
		}

		value *= r2.getMat().bsdf.f(bRec);

		//remove pixel differentials for further traversal as they no longer make sense
		bRec.dg.hasUVPartials = false;

		g_Image.Splat(dRec.uv.x, dRec.uv.y, value);
	}
}

CUDA_FUNC_IN void handleMediumInteraction(const Spectrum& weight, MediumSamplingRecord& mRec, const NormalizedT<Vec3f>& wi, const TraceResult& r2, Image& g_Image, CudaRNG& rng)
{
	DirectSamplingRecord dRec(mRec.p, NormalizedT<Vec3f>(0.0f));
	Spectrum value = weight * g_SceneData.sampleAttenuatedSensorDirect(dRec, rng.randomFloat2());
	if (!value.isZero() && V(dRec.p, dRec.ref))
	{
		value *= g_SceneData.m_sVolume.p(mRec.p, wi, (dRec.ref - dRec.p).normalized(), rng);
		if (!value.isZero())
			g_Image.Splat(dRec.uv.x, dRec.uv.y, value);
	}
} 

template<bool CORRECT_DIFFERENTIALS> CUDA_FUNC_IN void doWork(Image& g_Image, CudaRNG& rng)
{
	PositionSamplingRecord pRec;
	Spectrum power = g_SceneData.sampleEmitterPosition(pRec, rng.randomFloat2()), throughput = Spectrum(1.0f);

	handleEmission(power, pRec, g_Image, rng);

	DirectionSamplingRecord dRec;
	power *= ((const Light*)pRec.object)->sampleDirection(dRec, pRec, rng.randomFloat2());

	NormalizedT<Ray> r(pRec.p, dRec.d);
	TraceResult r2;
	r2.Init();
	int depth = -1;
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);

	KernelAggregateVolume& V = g_SceneData.m_sVolume;
	MediumSamplingRecord mRec;
	bool medium = false;
	const VolumeRegion* bssrdf = 0;

	while (++depth < 12 && !throughput.isZero())
	{
		TraceResult r2 = traceRay(r);
		float minT, maxT;
		if ((!bssrdf && V.HasVolumes() && V.IntersectP(r, 0, r2.m_fDist, &minT, &maxT) && V.sampleDistance(r, 0, r2.m_fDist, rng, mRec)) || (bssrdf && bssrdf->sampleDistance(r, 0, r2.m_fDist, rng.randomFloat(), mRec)))
		{
			throughput *= mRec.sigmaS * mRec.transmittance / mRec.pdfSuccess;
			handleMediumInteraction(power * throughput, mRec, -r.dir(), r2, g_Image, rng);
			if (bssrdf)
			{
				PhaseFunctionSamplingRecord pfRec(-r.dir());
				throughput *= bssrdf->As()->Func.Sample(pfRec, rng);
				r.dir() = pfRec.wi;
			}
			else throughput *= V.Sample(mRec.p, -r.dir(), rng, (NormalizedT<Vec3f>*)&r.dir());
			r.ori() = mRec.p;
			medium = true;
		}
		else if (!r2.hasHit())
			break;
		else
		{
			if (medium)
				throughput *= mRec.transmittance / mRec.pdfFailure;
			auto wo = bssrdf ? -r.dir() : r.dir();
			Spectrum f_i = power * throughput;
			r2.getBsdfSample(wo, r(r2.m_fDist), bRec, ETransportMode::EImportance, &rng, &f_i);
			handleSurfaceInteraction<false>(power * throughput, r2, bRec, r2, g_Image, rng);//CORRECT_DIFFERENTIALS
			Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
			if (f.isZero())
				break;
			if (!bssrdf && r2.getMat().GetBSSRDF(bRec.dg, &bssrdf))
				bRec.wo.z *= -1.0f;
			else
			{
				if (!bssrdf)
					throughput *= f;
				bssrdf = 0;
				medium = false;
			}

			if (depth > 5)
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

template<bool CORRECT_DIFFERENTIALS> __global__ void pathKernel(unsigned int N, Image g_Image)
{
	CudaRNG rng = g_RNGData();
	__shared__ volatile int nextRayArray[MaxBlockHeight];
	volatile int& rayBase = nextRayArray[threadIdx.y];
	do
	{
		if (threadIdx.x == 0)
			rayBase = atomicAdd(&g_NextRayCounter3, blockDim.x);

		int rayidx = rayBase + threadIdx.x;
		if (rayidx >= N)
			break;

		doWork<CORRECT_DIFFERENTIALS>(g_Image, rng);
	} while (true);
	g_RNGData(rng);
}

void PhotonTracer::DoRender(Image* I)
{
	unsigned int zero = 0;
	ThrowCudaErrors(cudaMemcpyToSymbol(g_NextRayCounter3, &zero, sizeof(unsigned int)));
	k_INITIALIZE(m_pScene, g_sRngs);
	if (m_sParameters.getValue(KEY_CorrectDifferentials()))
		pathKernel<true> << < 180, dim3(32, MaxBlockHeight, 1) >> >(w * h, *I);
	else pathKernel<false> << < 180, dim3(32, MaxBlockHeight, 1) >> >(w * h, *I);
	ThrowCudaErrors(cudaDeviceSynchronize());
}

void PhotonTracer::Debug(Image* I, const Vec2i& pixel)
{
	k_INITIALIZE(m_pScene, g_sRngs);
	CudaRNG rng = g_RNGData();
	doWork<true>(*I, rng);
	g_RNGData(rng);
}

}