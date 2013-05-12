#include "k_PrimTracer.h"
#include "k_TraceHelper.h"

__device__ float3 trace(Ray& r, CudaRNG& rng)
{
	TraceResult r2;
	r2.Init();
	float3 c = make_float3(1);
	unsigned int depth = 0;
	while(k_TraceRay<true>(r.direction, r.origin, &r2) && depth++ < 5)
	{//return make_float3(r.m_fDist/3);
		if(g_SceneData.m_sVolume.HasVolumes())
			c = c * exp(-g_SceneData.m_sVolume.tau(r, 0, r2.m_fDist));
		float3 wi;
		float pdf;
		e_KernelBSDF bsdf = r2.m_pTri->GetBSDF(r2.m_fUV, r2.m_pNode->getWorldMatrix(), g_SceneData.m_sMatData.Data, r2.m_pNode->m_uMaterialOffset);
		BxDFType sampledType;
		float3 f = bsdf.Sample_f(-r.direction, &wi, BSDFSample(0.64563f, 0.173f, rng.randomFloat()), &pdf, BSDF_ALL, &sampledType);
		f = bsdf.IntegratePdf(f, pdf, -r.direction);
		c = c * f;
		if((sampledType & BSDF_SPECULAR) != BSDF_SPECULAR)
			break;
		r.origin = r(r2.m_fDist);
		r.direction = wi;
		r2.Init();
	}
	c = c * r2.hasHit();
	return c;
}

__global__ void primaryKernel(long long width, long long height, RGBCOL* a_Data)
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
		float3 c = make_float3(0);
		float N = 1;
		for(float f = 0; f < N; f++)
		{
			Ray r = g_CameraData.GenRay(x, y, width, height, rng.randomFloat(), rng.randomFloat());
			c += trace(r, rng);
		}
		c /= N;
		/*
		Ray r = g_CameraData.GenRay(x, y, width, height, rng.randomFloat(), rng.randomFloat());
		TraceResult r2;
		r2.Init();
		k_TraceRay<true>(r.direction, r.origin, &r2);
		float3 c = make_float3(r2.m_fDist/length(g_SceneData.m_sBox.Size())*2.0f);*/

		unsigned int cl2 = toABGR(c);
		((unsigned int*)a_Data)[y * width + x] = cl2;
	}
	while(true);
	g_RNGData(rng);
}

__global__ void debugPixe2l(unsigned int width, unsigned int height, int2 p)
{
	Ray r = g_CameraData.GenRay(p.x, p.y, width, height);
	//dir = make_float3(-0.98181188f, 0.18984018f, -0.0024534566f);
	//ori = make_float3(68790.375f, -12297.199f, 57510.383f);
	//ori += make_float3(g_SceneData.m_sTerrain.m_sMin.x, 0, g_SceneData.m_sTerrain.m_sMin.z);
	trace(r, g_RNGData());
}

void k_PrimTracer::DoRender(RGBCOL* a_Buf)
{
	m_sRngs.m_uOffset++;
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, m_sRngs);
	primaryKernel<<< 180, dim3(32, MaxBlockHeight, 1)>>>(w, h, a_Buf);
	cudaError_t r = cudaThreadSynchronize();
	m_uRaysTraced = w * h;
	m_uPassesDone = 1;
}

void k_PrimTracer::Debug(int2 pixel)
{
	m_pScene->UpdateInvalidated();
	e_KernelDynamicScene d2 = m_pScene->getKernelSceneData();
	k_INITIALIZE(d2);
	k_STARTPASS(m_pScene, m_pCamera, m_sRngs);
	debugPixe2l<<<1,1>>>(w,h,pixel);
}