#include "k_FastTracer.h"
#include "k_TraceHelper.h"
#include "k_TraceAlgorithms.h"

namespace{
struct cameraData
{
	float2 m_InverseResolution;
	float4x4 m_sampleToCameraToWorld;
	float3 m_Position;
	cameraData(const e_PerspectiveCamera* c)
	{
		m_InverseResolution = c->m_invResolution;
		m_sampleToCameraToWorld = c->m_sampleToCamera * c->toWorld;
		m_Position = c->toWorld.Translation();
	}
};

CUDA_ALIGN(16) CUDA_DEVICE unsigned int g_NextRayCounter;
}


__global__ void pathCreateKernel(unsigned int width, unsigned int height, k_FastTracer::rayData* a_RayBuffer, TraceResult* a_ResBuffer, cameraData cData)
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
		
		//float3 q = cData.m_sampleToCameraToWorld * make_float3(cData.m_InverseResolution * make_float2(x, y), 0);
		//a_RayBuffer[rayidx].r = Ray(cData.m_Position, q - cData.m_Position);
		g_CameraData.sampleRay(a_RayBuffer[rayidx].r, make_float2(x,y),rng.randomFloat2());
		a_RayBuffer[rayidx].throughput = Spectrum(1.0f);
		a_ResBuffer[rayidx].m_fDist = FLT_MAX;
		a_RayBuffer[rayidx].x = x;
		a_RayBuffer[rayidx].y = y;
	}
	while(true);
	g_RNGData(rng);
}

CUDA_ALIGN(16) CUDA_DEVICE unsigned int g_NextInsertCounter;

__global__ void pathIterateKernel(unsigned int N, k_FastTracer::rayData* a_RayBuffer, TraceResult* a_ResBuffer, e_DirectImage g_Image, bool lastPass)
{
	CudaRNG rng = g_RNGData();
	int rayidx;
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

		k_FastTracer::rayData r = a_RayBuffer[rayidx];
		TraceResult r2 = a_ResBuffer[rayidx];
		if(r2.hasHit())
		{
			g_Image.SetPixel(r.x, r.y, Spectrum(r2.m_fDist/length(g_SceneData.m_sBox.Size())));
			continue;
		}else continue;

		BSDFSamplingRecord bRec;
		r2.getBsdfSample(r.r, rng, &bRec);
		Spectrum bsdfWeight = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
		//DirectSamplingRecord dRec(bRec.map.P, bRec.ng, bRec.map.uv);
		//g_Image.SetPixel(r.x,r.y, bsdfWeight * g_SceneData.sampleEmitterDirect(dRec, rng.randomFloat2()));
		
		r.L += r2.Le(bRec.map.P, bRec.map.sys, -r.r.direction) * r.throughput;
		r.throughput *= bsdfWeight;
		a_ResBuffer[rayidx].Init();
		r.throughput *= bsdfWeight;
		r.r.origin = bRec.map.P;
		r.r.direction = bRec.getOutgoing();
		unsigned int id = atomicInc(&g_NextInsertCounter, -1);
		//r.L += UniformSampleAllLights(bRec, r2.getMat(), 1);
		a_RayBuffer[id] = r;
		if(lastPass)
		{
			g_Image.SetPixel(r.x, r.y, r.L);
		}

	}
	while(true);
	g_RNGData(rng);
}

void k_FastTracer::doDirect(e_Image* I)
{
	k_ProgressiveTracer::DoRender(I);
	unsigned int zero = 0;
	intersector->ClearRays();
	intersector->ClearResults();
	cudaMemcpyToSymbol(g_NextRayCounter, &zero, sizeof(unsigned int));
	cudaMemcpyToSymbol(g_NextInsertCounter, &zero, sizeof(unsigned int));
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, g_sRngs);
	pathCreateKernel<<< 180, dim3(32, MaxBlockHeight, 1)>>>(w, h, intersector->getRayBuffer(), intersector->getResultBuffer(), cameraData(m_pCamera->As<e_PerspectiveCamera>()));
	intersector->IntersectBuffers(w * h);
	cudaMemcpyToSymbol(g_NextRayCounter, &zero, sizeof(unsigned int));
	cudaMemcpyToSymbol(g_NextInsertCounter, &zero, sizeof(unsigned int));
	pathIterateKernel<<< 180, dim3(32, MaxBlockHeight, 1)>>>(w * h, intersector->getRayBuffer(), intersector->getResultBuffer(), I->CreateDirectImage(), 1);
	m_uPassesDone++;
	m_uNumRaysTraced = w * h;
}

void k_FastTracer::doPath(e_Image* I)
{
	I->StartNewRendering();
	k_ProgressiveTracer::DoRender(I);
	unsigned int zero = 0;
	intersector->ClearRays();
	intersector->ClearResults();
	cudaMemcpyToSymbol(g_NextRayCounter, &zero, sizeof(unsigned int));
	cudaMemcpyToSymbol(g_NextInsertCounter, &zero, sizeof(unsigned int));
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, g_sRngs);
	pathCreateKernel<<< 180, dim3(32, MaxBlockHeight, 1)>>>(w, h, intersector->getRayBuffer(), intersector->getResultBuffer(), cameraData(m_pCamera->As<e_PerspectiveCamera>()));
	int raysToTrace = w * h, pass = 0;
	m_uNumRaysTraced = 0;
	e_DiffuseLight* LA = g_SceneData.m_sLightData[0].As<e_DiffuseLight>();
	do
	{
		intersector->IntersectBuffers(raysToTrace);
		cudaMemcpyToSymbol(g_NextRayCounter, &zero, sizeof(unsigned int));
		cudaMemcpyToSymbol(g_NextInsertCounter, &zero, sizeof(unsigned int));
		pathIterateKernel<<< 180, dim3(32, MaxBlockHeight, 1)>>>(raysToTrace, intersector->getRayBuffer(), intersector->getResultBuffer(), I->CreateDirectImage(), pass == 5);
		m_uNumRaysTraced += raysToTrace;
		cudaMemcpyFromSymbol(&raysToTrace, g_NextInsertCounter, sizeof(unsigned int));
	}
	while(raysToTrace && pass++ < 5);
	m_uPassesDone++;
}

void k_FastTracer::DoRender(e_Image* I)
{
	doPath(I);
}

void k_FastTracer::Resize(unsigned int w, unsigned int h)
{
	k_ProgressiveTracer::Resize(w, h);
	if(intersector)
		intersector->Free();
	intersector = new k_RayIntersectKernel<rayData>(w * h, 0);
}