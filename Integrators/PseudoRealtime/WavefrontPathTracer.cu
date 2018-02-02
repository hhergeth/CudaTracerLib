#include "WavefrontPathTracer.h"
#include <Math/Compression.h>
#include <Kernel/TraceAlgorithms.h>
#include <SceneTypes/Light.h>

namespace CudaTracerLib {

enum
{
	MaxBlockHeight = 6,
};

CUDA_DEVICE int g_NextRayCounterWPT;
CUDA_DEVICE CudaStaticWrapper<WavefrontPathTracerBuffer> g_ray_buffer;
CUDA_DEVICE DeviceDepthImage g_DepthImageWPT;

__global__ void pathCreateKernelWPT(unsigned int w, unsigned int h, BlockSamplerBuffer blockBuf)
{
	__shared__ volatile int nextRayArray[MaxBlockHeight];
	const int tidx = threadIdx.x;
	volatile int& rayBase = nextRayArray[threadIdx.y];
	do
	{
		if (tidx == 0)
			rayBase = atomicAdd(&g_NextRayCounterWPT, blockDim.x);

		int rayidx = rayBase + tidx;
		if (rayidx >= w * h)
			break;

		int x = rayidx % w, y = rayidx / w;
		unsigned int numSamples = blockBuf.getNumSamplesPerPixel(x, y);
		auto rng = g_SamplerData(rayidx);

		for (unsigned int i = 0; i < numSamples; i++)
		{
			NormalizedT<Ray> r;
			Spectrum W = g_SceneData.sampleSensorRay(r, Vec2f(x, y) + rng.randomFloat2(), rng.randomFloat2());
			WavefrontPTRayData dat;
			dat.x = half((float)x);
			dat.y = half((float)y);
			dat.throughput = W;
			dat.L = Spectrum(0.0f);
			dat.dIdx = UINT_MAX;
			dat.specular_bounce = true;
			g_ray_buffer->insertPayloadElement(dat, r);
		}
	} while (true);
}

template<bool NEXT_EVENT_EST> __global__ void pathIterateKernel(Image I, int pathDepth, int iterationIdx, int maxPathDepth, int RRStartDepth, bool depthImage)
{
	WavefrontPTRayData payload;
	NormalizedT<Ray> ray;
	TraceResult res;
	unsigned int rayIdx;
	while (g_ray_buffer->tryFetchPayloadElement(payload, ray, res, &rayIdx))
	{
		auto rng = g_SamplerData(rayIdx);
		rng.skip(iterationIdx + 2);//plus the camera sample

		if (NEXT_EVENT_EST && pathDepth > 0 && payload.dIdx != UINT_MAX)
		{
			traversalRay shadow_ray;
			traversalResult shadow_ray_res;
			if (g_ray_buffer->accessSecondaryRay(payload.dIdx, shadow_ray, shadow_ray_res))
			{
				if (shadow_ray_res.dist >= payload.dDist * (1 - 0.01f))
					payload.L += payload.directF;
			}
			payload.dIdx = UINT_MAX;
			payload.directF = 0.0f;
		}

		if (pathDepth == 0 && depthImage)
			g_DepthImageWPT.Store((int)payload.x.ToFloat(), (int)payload.y.ToFloat(), res.m_fDist);

		//if true the contribution will be added at the end of the loop body
		bool path_terminated = (pathDepth + 1 == maxPathDepth);

		if (res.hasHit())
		{
			BSDFSamplingRecord bRec;
			res.getBsdfSample(ray, bRec, ETransportMode::ERadiance);

			//account for emission
			if (res.LightIndex() != UINT_MAX)
			{
				float misWeight = 1.0f;
				if (!NEXT_EVENT_EST || pathDepth == 0 || payload.specular_bounce)
					misWeight = 1.0f;
				else
				{
					DirectSamplingRecord dRec = DirectSamplingRecFromRay(ray, res.m_fDist, Uchar2ToNormalizedFloat3((unsigned short)payload.prev_normal), bRec.dg.P, bRec.dg.n);
					auto* light = g_SceneData.getLight(res);
					float direct_pdf = light->pdfDirect(dRec) * g_SceneData.pdfEmitter(light);
					misWeight = MonteCarlo::PowerHeuristic(1, payload.bsdf_pdf, 1, direct_pdf);
				}
				payload.L += misWeight * res.Le(bRec.dg.P, bRec.dg.sys, -ray.dir()) * payload.throughput;
			}

			//do russian roulette
			bool surviveRR = true;
			if (pathDepth >= RRStartDepth)
			{
				if (rng.randomFloat() < payload.throughput.max())
					payload.throughput /= payload.throughput.max();
				else surviveRR = false;
			}

			if (pathDepth + 1 != maxPathDepth && surviveRR)
			{
				Spectrum f = res.getMat().bsdf.sample(bRec, payload.bsdf_pdf, rng.randomFloat2());
				payload.specular_bounce = (bRec.sampledType & EDelta) != 0;
				auto r_refl = NormalizedT<Ray>(bRec.dg.P, bRec.getOutgoing());

				payload.dIdx = UINT_MAX;
				if (NEXT_EVENT_EST && res.getMat().bsdf.hasComponent(ESmooth))
				{
					DirectSamplingRecord dRec(bRec.dg.P, bRec.dg.sys.n);
					Spectrum value = g_SceneData.sampleEmitterDirect(dRec, rng.randomFloat2());
					if (!value.isZero())
					{
						bRec.typeMask = EBSDFType(EAll & ~EDelta);
						bRec.wo = bRec.dg.toLocal(dRec.d);
						Spectrum bsdfVal = res.getMat().bsdf.f(bRec);
						const float bsdfPdf = res.getMat().bsdf.pdf(bRec);
						const float directPdf = dRec.measure == EArea ? PdfAtoW(dRec.pdf, dRec.dist, dot(dRec.n, dRec.d)) : dRec.pdf;
						const float weight = MonteCarlo::PowerHeuristic(1, directPdf, 1, bsdfPdf);
						payload.directF = payload.throughput * value * bsdfVal * weight;
						payload.dDist = dRec.dist;
						if (!g_ray_buffer->insertSecondaryRay(NormalizedT<Ray>(bRec.dg.P, dRec.d), payload.dIdx))
							payload.dIdx = UINT_MAX;
					}
				}

				payload.prev_normal = NormalizedFloat3ToUchar2(bRec.dg.sys.n);
				payload.throughput *= f;
				g_ray_buffer->insertPayloadElement(payload, r_refl);
			}
			else path_terminated = true;
		}
		else
		{
			path_terminated = true;
			float misWeight = 1.0f;
			if (!NEXT_EVENT_EST || pathDepth == 0 || payload.specular_bounce)
				misWeight = 1.0f;
			else if(g_SceneData.getEnvironmentMap() != 0)
			{
				DirectSamplingRecord dRec = DirectSamplingRecFromRay(ray, res.m_fDist, Uchar2ToNormalizedFloat3((unsigned short)payload.prev_normal), Vec3f(), NormalizedT<Vec3f>());
				auto* light = g_SceneData.getEnvironmentMap();
				float direct_pdf = light->pdfDirect(dRec) * g_SceneData.pdfEmitter(light);
				misWeight = MonteCarlo::PowerHeuristic(1, payload.bsdf_pdf, 1, direct_pdf);
			}
			payload.L += misWeight * payload.throughput * g_SceneData.EvalEnvironment(ray);
		}

		if (path_terminated)
		{
			I.AddSample(payload.x.ToFloat(), payload.y.ToFloat(), payload.L);
		}
	}
}

void WavefrontPathTracer::DoRender(Image* I)
{
	m_blockBuffer.Update(getBlockSampler());

	int maxPathLength = m_sParameters.getValue(KEY_MaxPathLength()), rrStart = m_sParameters.getValue(KEY_RRStartDepth());

	if (hasDepthBuffer())
		CopyToSymbol(g_DepthImageWPT, getDeviceDepthBuffer());
	m_ray_buf->StartFrame(g_SceneData.m_rayTraceEps);
	CopyToSymbol(g_ray_buffer, *m_ray_buf);
	ZeroSymbol(g_NextRayCounterWPT);
	pathCreateKernelWPT << < dim3(180, 1, 1), dim3(32, 6, 1) >> >(w, h, m_blockBuffer);
	CopyFromSymbol(*m_ray_buf, g_ray_buffer);

	int pass = 0;
	do
	{
		m_ray_buf->FinishIteration();
		CopyToSymbol(g_ray_buffer, *m_ray_buf);
		if (m_sParameters.getValue(KEY_Direct()))
			pathIterateKernel<true> << < dim3(180, 1, 1), dim3(32, 6, 1) >> >(*I, pass, m_uPassesDone, maxPathLength, rrStart, hasDepthBuffer());
		else pathIterateKernel<false> << < dim3(180, 1, 1), dim3(32, 6, 1) >> >(*I, pass, m_uPassesDone, maxPathLength, rrStart, hasDepthBuffer());
		CopyFromSymbol(*m_ray_buf, g_ray_buffer);
	} while (!m_ray_buf->isEmpty() && ++pass < maxPathLength);
	ThrowCudaErrors(cudaDeviceSynchronize());
}

}