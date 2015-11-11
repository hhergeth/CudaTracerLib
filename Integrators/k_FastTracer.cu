#include "k_FastTracer.h"
#include <Kernel/k_TraceHelper.h>
#include <Kernel/k_TraceAlgorithms.h>
#include <Engine/e_DynamicScene.h>

namespace CudaTracerLib {

CUDA_DEVICE e_Image g_DepthImage;

enum
{
	MaxBlockHeight = 6,
};

CUDA_DEVICE int g_NextRayCounterFT;
__global__ void pathCreateKernel(unsigned int w, unsigned int h, k_PTDBuffer g_Intersector)
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
				rayBase = atomicAdd(&g_NextRayCounterFT, numTerminated);

			rayidx = rayBase + idxTerminated;
			if (rayidx >= w * h)
				break;
		}

		int x = rayidx % w, y = rayidx / w;
		Ray r;
		Spectrum W = g_SceneData.sampleSensorRay(r, Vec2f(x, y), Vec2f(0, 0));
		traversalRay& ray = g_Intersector(rayidx, 0);
		ray.a = Vec4f(r.origin, 0.0f);
		ray.b = Vec4f(r.direction, FLT_MAX);
		rayData& dat = g_Intersector(rayidx);
		dat.x = x;
		dat.y = y;
		dat.throughput = W;
		dat.L = Spectrum(0.0f);
		dat.dIdx = UINT_MAX;
	} while (true);
}

__global__ void doDirectKernel(unsigned int w, unsigned int h, k_PTDBuffer g_Intersector, e_Image I, float SCALE, bool depthImage)
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
				rayBase = atomicAdd(&g_NextRayCounterFT, numTerminated);

			rayidx = rayBase + idxTerminated;
			if (rayidx >= w * h)
				break;
		}

		traversalResult& res = g_Intersector.res(rayidx, 0);
		RGBCOL col;
		col.x = col.w = 255;
		col.y = col.z = 0;
		float d = 1.0f;
		if (res.dist)
		{
			//tar[rayidx] = Spectrum(a_ResBuffer[rayidx].m_fDist/SCALE).toRGBCOL();
			float f = res.dist / SCALE * 255.0f;
			unsigned char c = (unsigned char)f;
			unsigned int i = (255 << 24) | (c << 16) | (c << 8) | c;
			col = *(RGBCOL*)&i;
			d = CalcZBufferDepth(g_SceneData.m_Camera.As()->m_fNearFarDepths.x, g_SceneData.m_Camera.As()->m_fNearFarDepths.y, res.dist);
		}
		Spectrum s;
		s.fromRGBCOL(col);
		I.AddSample(rayidx % w, rayidx / w, s);
		if (depthImage)
			g_DepthImage.SetSample(rayidx % w, rayidx / w, *(RGBCOL*)&d);
	} while (true);
}

__device__ CUDA_INLINE Vec2f stratifiedSample(const Vec2f& f, int pass)
{
	return f;
	//int i = pass % 64;
	//int x = i % 8, y = i / 8;
	//return Vec2f(x, y) / 8.0f + f / 8.0f;
}

#define max_PASS 7
CUDA_DEVICE k_PTDBuffer g_Intersector;
CUDA_DEVICE k_PTDBuffer g_Intersector2;
//CUDA_CONST e_KernelMaterial* g_MAT;
template<bool NEXT_EVENT_EST> __global__ void pathIterateKernel(unsigned int N, e_Image I, int pass, int iterationIdx, bool depthImage)
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
				rayBase = atomicAdd(&g_NextRayCounterFT, numTerminated);

			rayidx = rayBase + idxTerminated;
			if (rayidx >= N)
				break;
		}

		rayData dat = g_Intersector(rayidx);
		if (NEXT_EVENT_EST && pass > 0 && dat.dIdx != UINT_MAX)
		{
			traversalResult& res = g_Intersector.res(dat.dIdx, 1);
			traversalRay& ray = g_Intersector(dat.dIdx, 1);
			if (res.dist >= dat.dDist * 0.95f)
				dat.L += dat.directF;
		}

		traversalResult& res = g_Intersector.res(rayidx, 0);
		traversalRay& ray = g_Intersector(rayidx, 0);

		if (pass == 0)
		{
			float d = 1;
			if (res.dist)
				d = CalcZBufferDepth(g_SceneData.m_Camera.As()->m_fNearFarDepths.x, g_SceneData.m_Camera.As()->m_fNearFarDepths.y, res.dist);
			if (depthImage)
				g_DepthImage.SetSample(dat.x, dat.y, *(RGBCOL*)&d);
		}

		if (res.dist)
		{
			Ray r(ray.a.getXYZ(), ray.b.getXYZ());
			TraceResult r2;
			res.toResult(&r2, g_SceneData);
			DifferentialGeometry dg;
			BSDFSamplingRecord bRec(dg);
			r2.getBsdfSample(r, bRec, ETransportMode::ERadiance, &rng);
			if (pass == 0 || dat.dIdx == UINT_MAX)
				dat.L += r2.Le(bRec.dg.P, bRec.dg.sys, -r.direction) * dat.throughput;
			if (pass + 1 != max_PASS)
			{
				Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
				if (pass > 3)
				{
					if (rng.randomFloat() < f.max())
						f = f / f.max();
					else goto labelAdd;
				}
				unsigned int idx2 = g_Intersector2.insertRay(0);
				traversalRay& ray2 = g_Intersector2(idx2, 0);
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
					dat.dIdx = g_Intersector2.insertRay(1);
					traversalRay& ray3 = g_Intersector2(dat.dIdx, 1);
					ray3.a = Vec4f(bRec.dg.P, 1e-2f);
					ray3.b = Vec4f(dRec.d, FLT_MAX);
				}
				else dat.dIdx = UINT_MAX;
				dat.throughput *= f;
				g_Intersector2(idx2) = dat;
			}
		}
		else dat.L += dat.throughput * g_SceneData.EvalEnvironment(Ray(ray.a.getXYZ(), ray.b.getXYZ()));

		if (!res.dist || pass + 1 == max_PASS)
		{
		labelAdd:
			I.AddSample(dat.x, dat.y, dat.L);
		}
	} while (true);
	g_RNGData(rng);
}

static int zero = 0;

void k_FastTracer::doDirect(e_Image* I)
{
	k_PTDBuffer* buf = bufA;
	float scl = length(g_SceneData.m_sBox.Size());
	cudaMemcpyToSymbol(g_NextRayCounterFT, &zero, sizeof(zero));
	pathCreateKernel << < dim3(180, 1, 1), dim3(32, 6, 1) >> >(w, h, *buf);
	buf->setNumRays(w * h, 0);

	buf->IntersectBuffers<false>(false);

	I->Clear();
	cudaMemcpyToSymbol(g_NextRayCounterFT, &zero, sizeof(zero));
	doDirectKernel << < dim3(180, 1, 1), dim3(32, 6, 1) >> >(w, h, *buf, *I, scl, depthImage ? 1 : 0);
}

void k_FastTracer::doPath(e_Image* I)
{
	bufA->Clear();
	cudaMemcpyToSymbol(g_NextRayCounterFT, &zero, sizeof(zero));
	pathCreateKernel << < dim3(180, 1, 1), dim3(32, 6, 1) >> >(w, h, *bufA);
	bufA->setNumRays(w * h, 0);
	int pass = 0;
	k_PTDBuffer* srcBuf = bufA, *destBuf = bufB;
	size_t total, free;
	cudaMemGetInfo(&free, &total);
	do
	{
		destBuf->Clear();
		srcBuf->IntersectBuffers<false>(false);
		cudaMemcpyToSymbol(g_Intersector, srcBuf, sizeof(*srcBuf));
		cudaMemcpyToSymbol(g_Intersector2, destBuf, sizeof(*destBuf));
		cudaMemcpyToSymbol(g_NextRayCounterFT, &zero, sizeof(zero));
		if (m_pScene->getLightCount() > 2)
			pathIterateKernel<true> << < dim3(180, 1, 1), dim3(32, 6, 1) >> >(srcBuf->getNumRays(0), *I, pass, m_uPassesDone, depthImage ? 1 : 0);
		else pathIterateKernel<false> << < dim3(180, 1, 1), dim3(32, 6, 1) >> >(srcBuf->getNumRays(0), *I, pass, m_uPassesDone, depthImage ? 1 : 0);
		cudaMemcpyFromSymbol(srcBuf, g_Intersector, sizeof(*srcBuf));
		cudaMemcpyFromSymbol(destBuf, g_Intersector2, sizeof(*destBuf));
		swapk(srcBuf, destBuf);
	} while (srcBuf->getNumRays(0) && ++pass < max_PASS);
	cudaMemGetInfo(&free, &total);
}

void k_FastTracer::DoRender(e_Image* I)
{
	k_INITIALIZE(m_pScene, k_TracerBase::g_sRngs);
	if (depthImage)
	{
		cudaMemcpyToSymbol(g_DepthImage, depthImage, sizeof(e_Image));
		depthImage->StartRendering();
	}
	if (pathTracer)
		doPath(I);
	else doDirect(I);
	if (depthImage)
		depthImage->EndRendering();
}

}