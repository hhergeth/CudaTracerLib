#include "FastTracer.h"
#include <Kernel/TraceHelper.h>
#include <Kernel/TraceAlgorithms.h>
#include <Engine/DynamicScene.h>

namespace CudaTracerLib {

CUDA_DEVICE FastTracerBuffer g_IntersectorFT;
CUDA_DEVICE int g_NextRayCounterFT;

enum
{
	MaxBlockHeight = 6,
};

__global__ void pathCreateKernelFT(unsigned int w, unsigned int h)
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
		traversalRay& ray = g_IntersectorFT(rayidx, 0);
		ray.a = Vec4f(r.origin, 0.0f);
		ray.b = Vec4f(r.direction, FLT_MAX);
	} while (true);
}

__global__ void doDirectKernel(unsigned int w, unsigned int h, Image I, float SCALE, bool depthImage, DeviceDepthImage dImg)
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

		traversalResult& res = g_IntersectorFT.res(rayidx, 0);
		RGBCOL col;
		col.x = col.w = 255;
		col.y = col.z = 0;
		if (res.dist)
		{
			//tar[rayidx] = Spectrum(a_ResBuffer[rayidx].m_fDist/SCALE).toRGBCOL();
			float f = res.dist / SCALE * 255.0f;
			unsigned char c = (unsigned char)f;
			unsigned int i = (255 << 24) | (c << 16) | (c << 8) | c;
			col = *(RGBCOL*)&i;
		}
		Spectrum s;
		s.fromRGBCOL(col);
		I.AddSample(rayidx % w, rayidx / w, s);
		if (depthImage)
			dImg.Store(rayidx % w, rayidx / w, res.dist);
	} while (true);
}

void FastTracer::DoRender(Image* I)
{
	k_INITIALIZE(m_pScene, TracerBase::g_sRngs);
	ZeroSymbol(g_NextRayCounterFT);
	CopyToSymbol(g_IntersectorFT, *bufA);
	pathCreateKernelFT << < dim3(180, 1, 1), dim3(32, 6, 1) >> >(w, h);
	CopyFromSymbol(*bufA, g_IntersectorFT);
	bufA->setNumRays(w * h, 0);

	bufA->IntersectBuffers<false>(false);

	I->Clear();
	ZeroSymbol(g_NextRayCounterFT);
	CopyToSymbol(g_IntersectorFT, *bufA);
	float scl = length(g_SceneData.m_sBox.Size());
	doDirectKernel << < dim3(180, 1, 1), dim3(32, 6, 1) >> >(w, h, *I, scl, hasDepthBuffer(), getDeviceDepthBuffer());
}

}