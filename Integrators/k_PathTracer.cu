#include "k_PathTracer.h"
#include "..\Kernel\k_TraceHelper.h"
#include <time.h>
#include "..\Kernel\k_TraceAlgorithms.h"

CUDA_ALIGN(16) CUDA_DEVICE unsigned int g_NextRayCounter;

template<bool DIRECT> __global__ void pathKernel(unsigned int width, unsigned int height, unsigned int a_PassIndex, e_Image g_Image)
{
	CudaRNG rng = g_RNGData();
	int rayidx;
	int N = width * height;
	__shared__ volatile int nextRayArray[MaxBlockHeight];
	do
    {
        const int tidx = threadIdx.x;
        volatile int& rayBase = nextRayArray[threadIdx.y];

        const bool          terminated     = 1;//nodeAddr == EntrypointSentinel;
        const unsigned int  maskTerminated = __ballot(terminated);
        const int           numTerminated  = __popc(maskTerminated);
        const int           idxTerminated  = __popc(maskTerminated & ((1u<<tidx)-1));	

        if(terminated)
        {			
            if (idxTerminated == 0)
				rayBase = atomicAdd(&g_NextRayCounter, numTerminated);

            rayidx = rayBase + idxTerminated;
			if (rayidx >= N)
                break;
		}

		unsigned int x = rayidx % width, y = rayidx / width;
		Ray r;
		Spectrum imp = g_SceneData.sampleSensorRay(r, make_float2(x, y), rng.randomFloat2());

		Spectrum col = imp * PathTrace<DIRECT>(r.direction, r.origin, rng);
		
		g_Image.AddSample(x, y, col);
	}
	while(true);
	g_RNGData(rng);
}

__global__ void debugPixel(unsigned int width, unsigned int height, int2 p)
{
	CudaRNG rng = g_RNGData();
	Ray r = g_SceneData.GenerateSensorRay(p.x, p.y);	
	PathTrace<true>(r.direction, r.origin, rng);
}

void k_PathTracer::DoRender(e_Image* I)
{
	k_ProgressiveTracer::DoRender(I);
	ZeroSymbol(g_NextRayCounter);
	k_INITIALIZE(m_pScene, g_sRngs);
	if(m_Direct)
		pathKernel<true><<< 180, dim3(32, MaxBlockHeight, 1)>>>(w, h, m_uPassesDone, *I);
	else pathKernel<false><<< 180, dim3(32, MaxBlockHeight, 1)>>>(w, h, m_uPassesDone, *I);
	m_uPassesDone++;
	k_TracerBase_update_TracedRays
	I->DoUpdateDisplay(m_uPassesDone);
}

void k_PathTracer::Debug(int2 p)
{
	k_INITIALIZE(m_pScene, g_sRngs);
	//debugPixel<<<1,1>>>(w,h,p);
	CudaRNG rng = g_RNGData();
	Ray r = g_SceneData.GenerateSensorRay(p.x, p.y);	
	PathTrace<true>(r.direction, r.origin, rng);
}

template<bool DIRECT> __global__ void pathKernel2(unsigned int width, unsigned int height, e_Image g_Image, k_BlockSampler sampler)
{
	uint2 pixel = sampler.pixelCoord();
	CudaRNG rng = g_RNGData();
	if(pixel.x < width && pixel.y < height)
	{
		Ray r;
		Spectrum imp = g_SceneData.sampleSensorRay(r, make_float2(pixel.x, pixel.y), rng.randomFloat2());
		Spectrum col = imp * PathTrace<DIRECT>(r.direction, r.origin, rng);
		g_Image.AddSample(pixel.x, pixel.y, col);
	}
	g_RNGData(rng);
}

void k_BlockPathTracer::DoRender(e_Image* I)
{
	k_ProgressiveTracer::DoRender(I);
	k_INITIALIZE(m_pScene, g_sRngs);
	if(m_Direct)
		pathKernel2<true><<< sampler.blockDim(), sampler.threadDim()>>>(w, h, *I, sampler);
	else pathKernel2<false><<< sampler.blockDim(), sampler.threadDim()>>>(w, h, *I, sampler);
	k_TracerBase_update_TracedRays
	I->DoUpdateDisplay(1);
	sampler.AddPass(*I);
}