#include "k_PathTracer.h"
#include "k_TraceHelper.h"
#include "k_IntegrateHelper.h"
#include <time.h>
#include "k_TraceAlgorithms.h"

CUDA_ALIGN(16) CUDA_DEVICE unsigned int g_NextRayCounter;

template<bool DIRECT> __global__ void pathKernel(unsigned int width, unsigned int height, unsigned int a_PassIndex)
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
		CameraSample s = nextSample(x, y, rng, true);
		Ray r = g_CameraData.GenRay(s, width, height);
		
		float3 col = PathTrace<DIRECT>(r.direction, r.origin, rng);
		
		g_Image.AddSample(s, col);
	}
	while(true);
	g_RNGData(rng);
}

template<bool DIRECT> __global__ void debugPixel(unsigned int width, unsigned int height, int2 p)
{
	CudaRNG rng = g_RNGData();
	CameraSample s = nextSample(p.x, p.y, rng);
	Ray r = g_CameraData.GenRay(s, width, height);
		
	PathTrace<DIRECT>(r.direction, r.origin, rng);
}

void k_PathTracer::DoRender(e_Image* I)
{
	k_ProgressiveTracer::DoRender(I);
	unsigned int zero = 0;
	cudaMemcpyToSymbol(g_NextRayCounter, &zero, sizeof(unsigned int));
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASSI(m_pScene, m_pCamera, g_sRngs, *I);
	if(m_Direct)
		pathKernel<true><<< 180, dim3(32, MaxBlockHeight, 1)>>>(w, h, m_uPassesDone);
	else pathKernel<false><<< 180, dim3(32, MaxBlockHeight, 1)>>>(w, h, m_uPassesDone);
	m_uPassesDone++;
	k_ENDPASSI(I)
	k_TracerBase_update_TracedRays
	if(m_uPassesDone % 5 == 0)
		I->UpdateDisplay();
}

void k_PathTracer::Debug(int2 pixel)
{
	m_pScene->UpdateInvalidated();
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, g_sRngs);
	debugPixel<true><<<1,1>>>(w,h,pixel);
}