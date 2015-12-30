#include "WavefrontPathTracer.h"

namespace CudaTracerLib {

enum
{
	MaxBlockHeight = 6,
};

CUDA_DEVICE int g_NextRayCounterWPT;
CUDA_DEVICE WavefrontPathTracerBuffer g_IntersectorWPT;
CUDA_DEVICE WavefrontPathTracerBuffer g_Intersector2WPT;
CUDA_DEVICE DeviceDepthImage g_DepthImageWPT;

__global__ void pathCreateKernelWPT(unsigned int w, unsigned int h)
{
	int rayidx;
	__shared__ volatile int nextRayArray[MaxBlockHeight];
	do
	{
		const int tidx = threadIdx.x;
		volatile int& rayBase = nextRayArray[threadIdx.y];

		const bool          terminated = 1;//nodeAddr == EntrypointSentinel;
		const unsigned int  maskTerminated = __ballot(terminated);
		const int           numTerminated = __popc(maskTerminated);
		const int           idxTerminated = __popc(maskTerminated & ((1u << tidx) - 1));

		if (terminated)
		{
			if (idxTerminated == 0)
				rayBase = atomicAdd(&g_NextRayCounterWPT, numTerminated);

			rayidx = rayBase + idxTerminated;
			if (rayidx >= w * h)
				break;
		}

		int x = rayidx % w, y = rayidx / w;
		Ray r;
		Spectrum W = g_SceneData.sampleSensorRay(r, Vec2f(x, y), Vec2f(0, 0));
		traversalRay& ray = g_IntersectorWPT(rayidx, 0);
		ray.a = Vec4f(r.origin, 0.0f);
		ray.b = Vec4f(r.direction, FLT_MAX);
		auto& dat = g_IntersectorWPT(rayidx);
		dat.x = x;
		dat.y = y;
		dat.throughput = W;
		dat.L = Spectrum(0.0f);
		dat.dIdx = UINT_MAX;
	} while (true);
}


CUDA_ONLY_FUNC Vec2f stratifiedSample(const Vec2f& f, int pass)
{
	return f;
	//int i = pass % 64;
	//int x = i % 8, y = i / 8;
	//return Vec2f(x, y) / 8.0f + f / 8.0f;
}

template<bool NEXT_EVENT_EST> __global__ void pathIterateKernel(unsigned int N, Image I, int pathDepth, int iterationIdx, int maxPathDepth, bool depthImage)
{
	CudaRNG rng = g_RNGData();
	int rayidx;
	__shared__ volatile int nextRayArray[MaxBlockHeight];
	do
	{
		const int tidx = threadIdx.x;
		volatile int& rayBase = nextRayArray[threadIdx.y];

		const bool          terminated = 1;//nodeAddr == EntrypointSentinel;
		const unsigned int  maskTerminated = __ballot(terminated);
		const int           numTerminated = __popc(maskTerminated);
		const int           idxTerminated = __popc(maskTerminated & ((1u << tidx) - 1));

		if (terminated)
		{
			if (idxTerminated == 0)
				rayBase = atomicAdd(&g_NextRayCounterWPT, numTerminated);

			rayidx = rayBase + idxTerminated;
			if (rayidx >= N)
				break;
		}

		auto dat = g_IntersectorWPT(rayidx);
		if (NEXT_EVENT_EST && pathDepth > 0 && dat.dIdx != UINT_MAX)
		{
			traversalResult& res = g_IntersectorWPT.res(dat.dIdx, 1);
			traversalRay& ray = g_IntersectorWPT(dat.dIdx, 1);
			if (res.dist >= dat.dDist * 0.95f)
				dat.L += dat.directF;
		}

		traversalResult& res = g_IntersectorWPT.res(rayidx, 0);
		traversalRay& ray = g_IntersectorWPT(rayidx, 0);

		if (pathDepth == 0 && depthImage)
			g_DepthImageWPT.Store((int)dat.x.ToFloat(), (int)dat.y.ToFloat(), res.dist);

		if (res.dist)
		{
			Ray r(ray.a.getXYZ(), ray.b.getXYZ());
			TraceResult r2;
			res.toResult(&r2, g_SceneData);
			DifferentialGeometry dg;
			BSDFSamplingRecord bRec(dg);
			r2.getBsdfSample(r, bRec, ETransportMode::ERadiance, &rng);
			if (pathDepth == 0 || dat.dIdx == UINT_MAX)
				dat.L += r2.Le(bRec.dg.P, bRec.dg.sys, -r.direction) * dat.throughput;
			if (pathDepth + 1 != maxPathDepth)
			{
				Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
				if (pathDepth > 3)
				{
					if (rng.randomFloat() < f.max())
						f = f / f.max();
					else goto labelAdd;
				}
				unsigned int idx2 = g_Intersector2WPT.insertRay(0);
				traversalRay& ray2 = g_Intersector2WPT(idx2, 0);
				ray2.a = Vec4f(bRec.dg.P, 1e-2f);
				ray2.b = Vec4f(bRec.getOutgoing(), FLT_MAX);

				DirectSamplingRecord dRec(bRec.dg.P, bRec.dg.sys.n);
				Spectrum value = g_SceneData.sampleEmitterDirect(dRec, rng.randomFloat2());
				if (NEXT_EVENT_EST && r2.getMat().bsdf.hasComponent(ESmooth) && !value.isZero())
				{
					bRec.wo = normalize(bRec.dg.toLocal(dRec.d));
					Spectrum bsdfVal = r2.getMat().bsdf.f(bRec);
					const float bsdfPdf = r2.getMat().bsdf.pdf(bRec);
					const float weight = MonteCarlo::PowerHeuristic(1, dRec.pdf, 1, bsdfPdf);
					dat.directF = dat.throughput * value * bsdfVal * weight;
					dat.dDist = dRec.dist;
					dat.dIdx = g_Intersector2WPT.insertRay(1);
					traversalRay& ray3 = g_Intersector2WPT(dat.dIdx, 1);
					ray3.a = Vec4f(bRec.dg.P, 1e-2f);
					ray3.b = Vec4f(dRec.d, FLT_MAX);
				}
				else dat.dIdx = UINT_MAX;
				dat.throughput *= f;
				g_Intersector2WPT(idx2) = dat;
			}
		}
		else dat.L += dat.throughput * g_SceneData.EvalEnvironment(Ray(ray.a.getXYZ(), ray.b.getXYZ()));

		if (!res.dist || pathDepth + 1 == maxPathDepth)
		{
		labelAdd:
			I.AddSample(dat.x.ToFloat(), dat.y.ToFloat(), dat.L);
		}
	} while (true);
	g_RNGData(rng);
}

void WavefrontPathTracer::DoRender(Image* I)
{
	if (hasDepthBuffer())
		CopyToSymbol(g_DepthImageWPT, getDeviceDepthBuffer());
	bufA->Clear();
	ZeroSymbol(g_NextRayCounterWPT);
	CopyToSymbol(g_IntersectorWPT, *bufA);
	pathCreateKernelWPT << < dim3(180, 1, 1), dim3(32, 6, 1) >> >(w, h);
	CopyFromSymbol(*bufA, g_IntersectorWPT);
	bufA->setNumRays(w * h, 0);
	int pass = 0, maxPathLength = m_sParameters.getValue(KEY_MaxPathLength());
	WavefrontPathTracerBuffer* srcBuf = bufA, *destBuf = bufB;
	do
	{
		destBuf->Clear();
		srcBuf->IntersectBuffers<false>(false);
		CopyToSymbol(g_IntersectorWPT, *srcBuf); CopyToSymbol(g_Intersector2WPT, *destBuf);
		ZeroSymbol(g_NextRayCounterWPT);
		if (m_sParameters.getValue(KEY_Direct()))
			pathIterateKernel<true> << < dim3(180, 1, 1), dim3(32, 6, 1) >> >(srcBuf->getNumRays(0), *I, pass, m_uPassesDone, maxPathLength, hasDepthBuffer());
		else pathIterateKernel<false> << < dim3(180, 1, 1), dim3(32, 6, 1) >> >(srcBuf->getNumRays(0), *I, pass, m_uPassesDone, maxPathLength, hasDepthBuffer());
		CopyFromSymbol(*srcBuf, g_IntersectorWPT); CopyFromSymbol(*destBuf, g_Intersector2WPT);
		swapk(srcBuf, destBuf);
	} while (srcBuf->getNumRays(0) && ++pass < maxPathLength);
}

}