#include "k_PathTracer.h"
#include "k_TraceHelper.h"
#include "k_IntegrateHelper.h"
#include <time.h>
#include "k_TraceAlgorithms.h"

__global__ void pathKernel(unsigned int width, unsigned int height, RGBCOL* a_Data, unsigned int a_PassIndex, float4* a_DataTmp)
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
		Ray r = g_CameraData.GenRay<true>(x, y, width, height,  rng.randomFloat(), rng.randomFloat());
		
		float3 col = PathTrace<true>(r.direction, r.origin, rng);
		
		float4* data = a_DataTmp + y * width + x;
		*data += make_float4(col, 0);
		a_Data[y * width + x] = Float3ToCOLORREF(clamp01(!*data / (float)a_PassIndex));
	}
	while(true);
	g_RNGData(rng);
}

__global__ void debugPixel(unsigned int width, unsigned int height, int2 p)
{
	CudaRNG rng = g_RNGData();
	Ray r = g_CameraData.GenRay<false>(p.x, p.y, width, height,  rng.randomFloat(), rng.randomFloat());
		
	PathTrace<true>(r.direction, r.origin, rng);
}

void k_PathTracer::DoRender(RGBCOL* a_Buf)
{
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, m_sRngs);
	pathKernel<<< 180, dim3(32, MaxBlockHeight, 1)>>>(w, h, a_Buf, m_uPassesDone, m_pTmpData);
	m_uPassesDone++;
}

void k_PathTracer::StartNewTrace(RGBCOL* a_Buf)
{
	cudaMemset(m_pTmpData, 0, w * h * sizeof(float4));
}

void k_PathTracer::Resize(unsigned int _w, unsigned int _h)
{
	k_TracerBase::Resize(_w, _h);
	if(m_pTmpData)
		cudaFree(m_pTmpData);
	cudaMalloc(&m_pTmpData, sizeof(float4) * w * h);
	cudaMemset(m_pTmpData, 0, w * h * sizeof(float4));
}

void k_PathTracer::Debug(int2 pixel)
{
	m_pScene->UpdateInvalidated();
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, m_sRngs);
	debugPixel<<<1,1>>>(w,h,pixel);
}