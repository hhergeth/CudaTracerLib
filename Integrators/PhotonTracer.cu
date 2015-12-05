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
		const KernelLight* emitter = (const KernelLight*)pRec.object;
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
			Ray r, rX, rY;
			g_SceneData.sampleSensorRay(r, rX, rY, dRec.uv, Vec2f(0));
			Vec3f oldWi = bRec.wi;
			r2.getBsdfSample(r, bRec, ETransportMode::EImportance, &rng, &dRec.d);
			bRec.dg.computePartials(r, rX, rY);
			bRec.wi = oldWi;
		}

		value *= r2.getMat().bsdf.f(bRec);

		//remove pixel differentials for further traversal as they no longer make sense
		bRec.dg.hasUVPartials = false;

		g_Image.Splat(dRec.uv.x, dRec.uv.y, value);
	}
}

CUDA_FUNC_IN void handleMediumInteraction(const Spectrum& weight, MediumSamplingRecord& mRec, const Vec3f& wi, const TraceResult& r2, Image& g_Image, CudaRNG& rng)
{
	DirectSamplingRecord dRec(mRec.p, Vec3f(0));
	Spectrum value = weight * g_SceneData.sampleAttenuatedSensorDirect(dRec, rng.randomFloat2());
	if (!value.isZero() && V(dRec.p, dRec.ref))
	{
		value *= g_SceneData.m_sVolume.p(mRec.p, wi, (dRec.ref - dRec.p).normalized(), rng);
		if (!value.isZero())
			g_Image.Splat(dRec.uv.x, dRec.uv.y, value);
	}
}

CUDA_FUNC_IN Vec3f refract(const Vec3f &wi, float cosThetaT, float eta)
{
	float scale = -(cosThetaT < 0 ? (1.0f / eta) : eta);
	return Vec3f(scale*wi.x, scale*wi.y, cosThetaT);
}
CUDA_FUNC_IN Vec3f reflect(const Vec3f &wi)
{
	return Vec3f(-wi.x, -wi.y, wi.z);
}
CUDA_FUNC_IN Spectrum sample(const Spectrum& s_, BSDFSamplingRecord& bRec, CudaRNG& rng)
{
	float w;
	Spectrum s = s_.SampleSpectrum(w, rng.randomFloat());
	for (int i = 0; i < 3; i++)
		if (s_[i] != 0)
			s[i] /= s_[i];

	Vec3f B(1.03961212f, 0.231792344f, 1.01046945f), C(6.00069867e-3f, 2.00179144e-2f, 1.03560653e2f);
	float w_mu = w / 1e3;
	float eta = math::safe_sqrt(1 + ((B * w_mu * w_mu) / (Vec3f(w_mu * w_mu) - C)).sum());
	//float eta = math::lerp(1.4f, 1.8f, (w - 300) / (600));
	//float eta = 1.5f;

	float cosThetaT;
	float F = MonteCarlo::fresnelDielectricExt(Frame::cosTheta(bRec.wi), cosThetaT, eta);
	Vec2f sample = rng.randomFloat2();
	if (sample.x <= F) {
		bRec.sampledType = EDeltaReflection;
		bRec.wo = reflect(bRec.wi);
		bRec.eta = 1.0f;

		return Spectrum(1.0f);
	}
	else {
		bRec.sampledType = EDeltaTransmission;
		bRec.wo = refract(bRec.wi, cosThetaT, eta);
		bRec.eta = cosThetaT < 0 ? eta : (1.0f / eta);

		float factor = (bRec.mode == ETransportMode::ERadiance) ? (cosThetaT < 0 ? (1.0f / eta) : eta) : 1.0f;

		return s * (factor * factor);
	}
}

//if (r2.getMat().bsdf.getTypeToken() != UINT_MAX)
//r2.getMat().bsdf.getTypeToken() == UINT_MAX ? sample(power * throughput, bRec, rng) : 

template<bool CORRECT_DIFFERENTIALS> CUDA_FUNC_IN void doWork(Image& g_Image, CudaRNG& rng)
{
	PositionSamplingRecord pRec;
	Spectrum power = g_SceneData.sampleEmitterPosition(pRec, rng.randomFloat2()), throughput = Spectrum(1.0f);

	handleEmission(power, pRec, g_Image, rng);

	DirectionSamplingRecord dRec;
	power *= ((const KernelLight*)pRec.object)->sampleDirection(dRec, pRec, rng.randomFloat2());

	Ray r(pRec.p, dRec.d);
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
			handleMediumInteraction(power * throughput, mRec, -r.direction, r2, g_Image, rng);
			if (bssrdf)
			{
				PhaseFunctionSamplingRecord mRec(-r.direction);
				throughput *= bssrdf->As()->Func.Sample(mRec, rng);
				r.direction = mRec.wi;
			}
			else throughput *= V.Sample(mRec.p, -r.direction, rng, &r.direction);
			r.origin = mRec.p;
			medium = true;
		}
		else if (!r2.hasHit())
			break;
		else
		{
			if (medium)
				throughput *= mRec.transmittance / mRec.pdfFailure;
			Vec3f wo = bssrdf ? -r.direction : r.direction;
			r2.getBsdfSample(wo, r(r2.m_fDist), bRec, ETransportMode::EImportance, &rng);
			handleSurfaceInteraction<CORRECT_DIFFERENTIALS>(power * throughput, r2, bRec, r2, g_Image, rng);
			Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
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

			r = Ray(bRec.dg.P, bRec.getOutgoing());
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