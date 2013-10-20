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
}

__global__ void pathCreateKernel(unsigned int w, unsigned int h, k_RayBuffer<k_FastTracer::rayData, 2> buf, cameraData cData)
{
    int rayidx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
	if(rayidx >= w * h)
		return;
	unsigned int x = rayidx % w, y = rayidx / w;
		
	//g_CameraData.sampleRay(a_RayBuffer[rayidx].r, make_float2(x,y),rng.randomFloat2());
	buf.getPayloadBuffer()[rayidx].throughput = Spectrum(1.0f);
	//a_RayBuffer[rayidx].a = make_float4(cData.m_Position, 0.0f);
	//a_RayBuffer[rayidx].b = make_float4(normalize(q - cData.m_Position), FLT_MAX);
	Ray r;
	g_CameraData.sampleRay(r, make_float2(x,y), make_float2(0,0));
	
	buf[0].getRayBuffer()[rayidx].a = make_float4(r.origin, 0.0f);
	buf[0].getRayBuffer()[rayidx].b = make_float4(r.direction, FLT_MAX);
	buf.getPayloadBuffer()[rayidx].x = x;
	buf.getPayloadBuffer()[rayidx].y = y;
}

CUDA_ALIGN(16) CUDA_DEVICE unsigned int g_NextInsertCounter;

__global__ void pathIterateKernel(unsigned int w, unsigned int h, k_RayBuffer<k_FastTracer::rayData, 2> buf, RGBCOL* tar, float SCALE)
{/*
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
	g_RNGData(rng);*/
    int rayidx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
	if(rayidx >= w * h)
		return;
	//TraceResult& r2 = a_ResBuffer[rayidx];
	//k_FastTracer::rayData& r = a_PayloadBuffer[rayidx];
	//if(r2.hasHit())
	{
		//tar[rayidx] = Spectrum(a_ResBuffer[rayidx].m_fDist/SCALE).toRGBCOL();
		float f = buf[0].getResultBuffer()[rayidx].dist/SCALE * 255.0f;
		unsigned char c = (unsigned char)f;
		unsigned int i = (255 << 24) | (c << 16) | (c << 8) | c;
		unsigned int* t = ((unsigned int*)tar) + rayidx;
		*t = i;
	}
}

__global__ void pathIterateKernel(unsigned int N, unsigned int w, k_RayBuffer<k_FastTracer::rayData, 2> buf, e_Image I, bool last)
{
    int rayidx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
	if(rayidx >= N)
		return;
	k_FastTracer::rayData& d = buf.getPayloadBuffer()[rayidx];
	Ray r;
	TraceResult r2;
	BSDFSamplingRecord bRec;
	if(buf[0].getResultBuffer()[rayidx].dist)
	{
		CudaRNG rng = g_RNGData();
		r.origin = !buf[0].getRayBuffer()[rayidx].a;
		r.direction = !buf[0].getRayBuffer()[rayidx].b;
		buf[0].getResultBuffer()[rayidx].toResult(&r2, g_SceneData);
		r2.getBsdfSample(r, rng, &bRec);
		Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
		d.L += r2.Le(bRec.map.P, bRec.map.sys, r.direction) * d.throughput;
		d.throughput *= f;
		unsigned int id = atomicInc(&g_NextInsertCounter, -1);
		buf[0].getRayBuffer()[id].a = make_float4(bRec.map.P, 0);
		buf[0].getRayBuffer()[id].b = make_float4(bRec.getOutgoing(), FLT_MAX);
		if(last)
			I.AddSample(d.x, d.y, d.L);//Spectrum(-dot(bRec.ng, r.direction))
		g_RNGData(rng);
	}
}

#include "..\Base\Timer.h"
static cTimer TT;
void k_FastTracer::doDirect(e_Image* I)
{
	k_ProgressiveTracer::DoRender(I);
	unsigned int zero = 0;
	intersector->ClearRays();
	intersector->ClearResults();
	cudaMemcpyToSymbol(g_NextInsertCounter, &zero, sizeof(unsigned int));
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, g_sRngs);
	float scl = length(g_SceneData.m_sBox.Size());
	pathCreateKernel<<< dim3((w*h)/(32*8)+1,1,1), dim3(32, 8, 1)>>>(w, h, *intersector, cameraData(m_pCamera->As<e_PerspectiveCamera>()));
	/*
	TT.StartTimer();
	Ray r;
	float2 ps, at = make_float2(0);
	for(int i = 0; i < w; i++)
		for(int j = 0; j < h; j++)
		{
			float4* t = (float4*)(hostRays + (j * w + i));
			ps.x = i;
			ps.y = j;
			m_pCamera->sampleRay(r, ps, at);
			t[0] = make_float4(r.origin, 0);
			t[1] = make_float4(r.direction, FLT_MAX);
		}
	cudaMemcpy(intersector->m_pRayBuffer, hostRays, sizeof(traversalRay) * w * h, cudaMemcpyHostToDevice);
	m_fTimeSpentRendering = (float)TT.EndTimer();*/
	
	cudaEventRecord(start, 0);
	intersector->IntersectBuffers(w * h, m_pScene->getNodeCount() == 1);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	//m_fTimeSpentRendering = elapsedTime * 1e-3f;
	
	cudaEventRecord(start, 0);
	I->StartNewRendering();
	pathIterateKernel<<< dim3((w*h)/(32*8)+1,1,1), dim3(32, 8, 1)>>>(w * h, w, *intersector, *I, 1);
	I->UpdateDisplay();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	m_fTimeSpentRendering = elapsedTime * 1e-3f;
	
	/*TT.StartTimer();
	I->StartNewRendering();
	CudaRNG rng = g_RNGData();
	TraceResult r2;
	BSDFSamplingRecord bRec;
	cudaMemcpy(hostResults, intersector->m_pResultBuffer, sizeof(traversalResult) * w * h, cudaMemcpyDeviceToHost);
	for(int i = 0; i < w; i++)
		for(int j = 0; j < h; j++)
		{
			int id = j * w + i;
			if(hostResults[id].dist)
			{
				r.origin = !hostRays[id].a;
				r.direction = !hostRays[id].b;
				hostResults[id].toResult(&r2, g_SceneData);
				//r2.getBsdfSample(r, rng, &bRec);
				I->AddSample(i,j,Spectrum(-dot(bRec.ng, r.direction)));
			}
		}
	g_RNGData(rng);
	I->UpdateDisplay();
	m_fTimeSpentRendering = (float)TT.EndTimer();*/

	//cudaMemcpyToSymbol(g_NextInsertCounter, &zero, sizeof(unsigned int));
	//pathIterateKernel<<< dim3((w*h)/(32*8)+1,1,1), dim3(32, 8, 1)>>>(w, h, intersector->getRayBuffer(), intersector->getPayloadBuffer(), intersector->getResultBuffer(), I->CreateDirectImage().target, scl);
	
	m_uPassesDone++;
	m_uNumRaysTraced = w * h;
	cudaEventRecord(start, 0);
}

void k_FastTracer::doPath(e_Image* I)
{
	//I->StartNewRendering();
	k_ProgressiveTracer::DoRender(I);
	unsigned int zero = 0;
	intersector->ClearRays();
	intersector->ClearResults();
	cudaMemcpyToSymbol(g_NextInsertCounter, &zero, sizeof(unsigned int));
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, g_sRngs);
	pathCreateKernel<<< dim3((w*h)/(32*8)+1,1,1), dim3(32, 8, 1)>>>(w, h, *intersector, cameraData(m_pCamera->As<e_PerspectiveCamera>()));
	int raysToTrace = w * h, pass = 0;
	m_uNumRaysTraced = 0;
	const int depth = 5;
	do
	{
		intersector->ClearResults();
		intersector->IntersectBuffers(raysToTrace);
		cudaMemcpyToSymbol(g_NextInsertCounter, &zero, sizeof(unsigned int));
		pathIterateKernel<<< dim3((w*h)/(32*8)+1,1,1), dim3(32, 8, 1)>>>(w * h, w, *intersector, *I, pass == depth);
		m_uNumRaysTraced += raysToTrace;
		cudaMemcpyFromSymbol(&raysToTrace, g_NextInsertCounter, sizeof(unsigned int));
	}
	while(raysToTrace && pass++ < depth);
	m_uPassesDone++;
	I->UpdateDisplay();
}

void k_FastTracer::DoRender(e_Image* I)
{
	doPath(I);
}
/*
void k_FastTracer::Resize(unsigned int w, unsigned int h)
{
	k_ProgressiveTracer::Resize(w, h);
	if(hostRays)
		cudaFreeHost(hostRays);
	if(hostResults)
		cudaFreeHost(hostResults);
	cudaMallocHost(&hostRays, sizeof(traversalRay) * w * h);
	cudaMallocHost(&hostResults, sizeof(traversalResult) * w * h);
	if(intersector)
		intersector->Free();
	intersector = new k_RayBuffer<rayData, 2>(w * h);
}*/