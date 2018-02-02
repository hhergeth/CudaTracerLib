#include "FastTracer.h"
#include <Kernel/TraceHelper.h>
#include <Kernel/TraceAlgorithms.h>
#include <Engine/DynamicScene.h>

namespace CudaTracerLib {

CUDA_DEVICE CudaStaticWrapper<FastTracerBuffer> g_primary_ray_buffer;
CUDA_DEVICE int g_NextRayCounterFT;

enum
{
	MaxBlockHeight = 6,
};

__global__ void pathCreateKernelFT(unsigned int w, unsigned int h)
{
	__shared__ volatile int nextRayArray[MaxBlockHeight];
	const int tidx = threadIdx.x;
	volatile int& rayBase = nextRayArray[threadIdx.y];
	do
	{
		if (tidx == 0)
			rayBase = atomicAdd(&g_NextRayCounterFT, blockDim.x);

		int rayidx = rayBase + tidx;
		if (rayidx >= w * h)
			break;

		int x = rayidx % w, y = rayidx / w;
		NormalizedT<Ray> r;
		Spectrum W = g_SceneData.sampleSensorRay(r, Vec2f(x, y), Vec2f(0, 0));
		g_primary_ray_buffer->insertPayloadElement({(unsigned short)x, (unsigned short)y}, r);
	} while (true);
}

__global__ void doDirectKernel(unsigned int w, unsigned int h, Image I, float SCALE, bool depthImage, DeviceDepthImage dImg)
{
	EmptyRayData payload;
	NormalizedT<Ray> ray;
	TraceResult res;
	unsigned int rayIdx;
	while (g_primary_ray_buffer->tryFetchPayloadElement(payload, ray, res))
	{
		Spectrum s = 0.0f;
		if (res.hasHit())
			s = Spectrum(res.m_fDist / SCALE);
		I.AddSample(payload.x, payload.y, s);
		if (depthImage)
			dImg.Store(rayIdx % w, rayIdx / w, res.m_fDist);
	}
}

void FastTracer::DoRender(Image* I)
{
	bufA->StartFrame(g_SceneData.m_rayTraceEps);

	ZeroSymbol(g_NextRayCounterFT);
	CopyToSymbol(g_primary_ray_buffer, *bufA);
	pathCreateKernelFT << < dim3(180, 1, 1), dim3(32, 6, 1) >> >(w, h);
	CopyFromSymbol(*bufA, g_primary_ray_buffer);

	bufA->FinishIteration();

	I->Clear();
	ZeroSymbol(g_NextRayCounterFT);
	CopyToSymbol(g_primary_ray_buffer, *bufA);
	float scl = length(g_SceneData.m_sBox.Size());
	doDirectKernel << < dim3(180, 1, 1), dim3(32, 6, 1) >> >(w, h, *I, scl, hasDepthBuffer(), getDeviceDepthBuffer());
}

}