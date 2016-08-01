#include "PhotonTracer.h"
#include <Kernel/TraceHelper.h>
#include <Kernel/TraceAlgorithms.h>
#include <Engine/Light.h>
#include <Engine/Sensor.h>
#include <Kernel/ParticleProcess.h>

namespace CudaTracerLib {

enum
{
	MaxBlockHeight = 6,
};

CUDA_DEVICE unsigned int g_NextRayCounter3;

template<bool CORRECT_DIFFERENTIALS> struct PhotonTracerParticleProcessHandler
{
	Image& g_Image;
	Sampler& rng;

	CUDA_FUNC_IN PhotonTracerParticleProcessHandler(Image& I, Sampler& r)
		: g_Image(I), rng(r)
	{
		
	}

	CUDA_FUNC_IN void handleEmission(const Spectrum& weight, const PositionSamplingRecord& pRec)
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

	CUDA_FUNC_IN void handleSurfaceInteraction(const Spectrum& weight, float accum_pdf, const Spectrum& last_f, const NormalizedT<Ray>& r, const TraceResult& r2, BSDFSamplingRecord& bRec, bool lastBssrdf)
	{
		DirectSamplingRecord dRec(bRec.dg.P, bRec.dg.sys.n);
		Spectrum value = weight * g_SceneData.sampleAttenuatedSensorDirect(dRec, rng.randomFloat2());
		if (!value.isZero() && V(dRec.p, dRec.ref))
		{
			bRec.wo = bRec.dg.toLocal(dRec.d);

			//compute pixel differentials
			if (false&&CORRECT_DIFFERENTIALS)
			{
				NormalizedT<Ray> r, rX, rY;
				g_SceneData.sampleSensorRay(r, rX, rY, dRec.uv, Vec2f(0));
				auto oldWi = bRec.wi;
				r2.getBsdfSample(r, bRec, ETransportMode::ERadiance, 0, &dRec.d);
				bRec.dg.computePartials(r, rX, rY);
				bRec.wi = oldWi;
			}

			if (!lastBssrdf)
				value *= r2.getMat().bsdf.f(bRec);
			else value *= 1.0f;

			//remove pixel differentials for further traversal as they no longer make sense
			bRec.dg.hasUVPartials = false;

			g_Image.Splat(dRec.uv.x, dRec.uv.y, value);
		}
	}

	CUDA_FUNC_IN void handleMediumSampling(const Spectrum& weight, float accum_pdf, const Spectrum& last_f, const NormalizedT<Ray>& r, const TraceResult& r2, const MediumSamplingRecord& mRec, bool sampleInMedium, const VolumeRegion* bssrdf)
	{
		
	}

	CUDA_FUNC_IN void handleMediumInteraction(const Spectrum& weight, float accum_pdf, const Spectrum& last_f, const MediumSamplingRecord& mRec, const NormalizedT<Vec3f>& wi, const TraceResult& r2, const VolumeRegion* bssrdf)
	{
		if (!bssrdf)
		{
			DirectSamplingRecord dRec(mRec.p, NormalizedT<Vec3f>(0.0f));
			Spectrum value = weight * g_SceneData.sampleAttenuatedSensorDirect(dRec, rng.randomFloat2());
			if (!value.isZero() && V(dRec.p, dRec.ref))
			{
				PhaseFunctionSamplingRecord pRec(wi, dRec.d);
				value *= g_SceneData.m_sVolume.p(mRec.p, pRec);
				if (!value.isZero())
					g_Image.Splat(dRec.uv.x, dRec.uv.y, value);
			}
		}
	}
};

template<bool CORRECT_DIFFERENTIALS> __global__ void pathKernel(unsigned int N, Image g_Image)
{
	auto rng = g_SamplerData();
	auto process = PhotonTracerParticleProcessHandler<CORRECT_DIFFERENTIALS>(g_Image, rng);
	__shared__ volatile int nextRayArray[MaxBlockHeight];
	volatile int& rayBase = nextRayArray[threadIdx.y], i = 0;
	do
	{
		if (threadIdx.x == 0)
			rayBase = atomicAdd(&g_NextRayCounter3, blockDim.x);

		int rayidx = rayBase + threadIdx.x;
		if (rayidx >= N)
			break;

		rng.StartSequence(rayidx);
		ParticleProcess(12, 7, rng, process);
	} while (true);
	g_SamplerData(rng);
}

void PhotonTracer::DoRender(Image* I)
{
	unsigned int zero = 0;
	ThrowCudaErrors(cudaMemcpyToSymbol(g_NextRayCounter3, &zero, sizeof(unsigned int)));
	if (m_sParameters.getValue(KEY_CorrectDifferentials()))
		pathKernel<true> << < 180, dim3(32, MaxBlockHeight, 1) >> >(w * h, *I);
	else pathKernel<false> << < 180, dim3(32, MaxBlockHeight, 1) >> >(w * h, *I);
	ThrowCudaErrors(cudaDeviceSynchronize());
}

void PhotonTracer::DebugInternal(Image* I, const Vec2i& pixel)
{
	auto rng = g_SamplerData();
	auto process = PhotonTracerParticleProcessHandler<false>(*I, rng);
	for (int i = 0; i < 1000; i++)
		ParticleProcess(12, 7, rng, process);
	g_SamplerData(rng);
}

}