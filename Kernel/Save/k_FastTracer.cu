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

template<bool INSERT> __global__ void pathCreateKernel(unsigned int w, unsigned int h, k_RayBuffer<k_FastTracer::rayData, 2> g_Intersector)
{
    int rayidx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
	if(rayidx >= w * h)
		return;
	int x = rayidx % w, y = rayidx / w;

	Ray r;
	g_CameraData.sampleRay(r, make_float2(x,y), make_float2(0,0));
	traversalRay* ray;
	if(INSERT)
		ray = g_Intersector[0].InsertRay(rayidx, rayidx);
	else ray = g_Intersector[0].InsertRay(rayidx);
	ray->a = make_float4(r.origin, 0.0f);
	ray->b = make_float4(r.direction, FLT_MAX);
	g_Intersector(rayidx).x = x;
	g_Intersector(rayidx).y = y;
	g_Intersector(rayidx).throughput = Spectrum(1.0f);
}

__global__ void pathIterateKernel(unsigned int w, unsigned int h, k_RayBuffer<k_FastTracer::rayData, 2> g_Intersector, e_Image I, float SCALE)
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
	traversalResult* res;
	g_Intersector[0].FetchRay(rayidx, &res);
	if(res->dist)
	{
		//tar[rayidx] = Spectrum(a_ResBuffer[rayidx].m_fDist/SCALE).toRGBCOL();
		float f = res->dist/SCALE * 255.0f;
		unsigned char c = (unsigned char)f;
		unsigned int i = (255 << 24) | (c << 16) | (c << 8) | c;
		I.SetSample(rayidx % w, rayidx / w, *(RGBCOL*)&i);
	}
}

#define MAX_PASS 5
__global__ void pathIterateKernel(unsigned int N, e_Image I, int pass, k_RayBuffer<k_FastTracer::rayData, 2> g_Intersector)//template
{
    int slotIdx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
	if(slotIdx >= N)
		return;
	unsigned int payloadIdx = g_Intersector[0].GetPayloadIndex(slotIdx);
	k_FastTracer::rayData& d = g_Intersector(payloadIdx);
	traversalResult* res;
	traversalRay* ray = g_Intersector[0].FetchRay(slotIdx, &res);
	if(res->dist)
	{
		//I.SetSample(d.x, d.y, Spectrum(res->dist / length(g_SceneData.m_sBox.Size())).toRGBCOL());return;
		
		CudaRNG rng = g_RNGData();
		Ray r(!ray->a, !ray->b);
		TraceResult r2;
		res->toResult(&r2, g_SceneData);
		BSDFSamplingRecord bRec;
		r2.getBsdfSample(r, rng, &bRec);
		
		//traversalResult* tr;
	//	g_Intersector[1].FetchRay(d.dIndex, &tr);
		//if(pass && ((tr->dist >= d.dDist) || (tr->dist == 0.0f)))
		//	d.L += d.D;
		//d.D = Spectrum(0.0f);

		d.L += r2.Le(bRec.map.P, bRec.map.sys, -r.direction) * d.throughput;
		Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
		d.throughput *= f;
		unsigned int rayIdx;
		ray = g_Intersector[0].InsertRay(payloadIdx, &rayIdx);
		ray->a = make_float4(bRec.map.P, 0);
		ray->b = make_float4(bRec.getOutgoing(), FLT_MAX);
		if(pass == MAX_PASS)
			I.AddSample(d.x, d.y, d.L);//Spectrum(-dot(bRec.ng, r.direction))
		
		/*if(pass != MAX_PASS)
		{
			DirectSamplingRecord dRec(bRec.map.P, bRec.map.sys.n, bRec.map.uv);
			Spectrum value = g_SceneData.sampleEmitterDirect(dRec, rng.randomFloat2());
			bRec.wo = bRec.map.sys.toLocal(dRec.d);
			Spectrum bsdfVal = r2.getMat().bsdf.f(bRec);
			const float bsdfPdf = r2.getMat().bsdf.pdf(bRec);
			const float weight = MonteCarlo::PowerHeuristic(1, dRec.pdf, 1, bsdfPdf);
			d.D = value * weight * bsdfVal * d.throughput;
			traversalRay* ray2 = g_Intersector[1].InsertRay(payloadIdx, &d.dIndex);
			ray2->a = make_float4(dRec.ref, 0);
			ray2->b = make_float4(dRec.d, FLT_MAX);
			d.dDist = dRec.dist;
		}*/

		g_RNGData(rng);
	}
	else I.AddSample(d.x, d.y, d.L);
}

#include "..\Base\Timer.h"
static cTimer TT;
void k_FastTracer::doDirect(e_Image* I)
{
	k_ProgressiveTracer::DoRender(I);
	unsigned int zero = 0;
	intersector->StartNewRendering();
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, g_sRngs);
	float scl = length(g_SceneData.m_sBox.Size());
	pathCreateKernel<true> <<< dim3((w*h)/(32*8)+1,1,1), dim3(32, 8, 1)>>>(w, h, *intersector);
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
	intersector->IntersectBuffers<false>(0, m_pScene->getNodeCount() == 1);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	m_fTimeSpentRendering = elapsedTime * 1e-3f;
		/*
	cudaEventRecord(start, 0);
	I->StartNewRendering();
	pathIterateKernel<<< dim3((w*h)/(32*8)+1,1,1), dim3(32, 8, 1)>>>(w * h, w, *intersector, *I, 1, 1);
	I->UpdateDisplay();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	m_fTimeSpentRendering = elapsedTime * 1e-3f;

	TT.StartTimer();
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

	pathIterateKernel<<< dim3((w*h)/(32*8)+1,1,1), dim3(32, 8, 1)>>>(w, h, *intersector, *I, scl);
	
	m_uPassesDone++;
	m_uNumRaysTraced = w * h;
	cudaEventRecord(start, 0);
}

void k_FastTracer::doPath(e_Image* I)
{
	k_ProgressiveTracer::DoRender(I);
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, g_sRngs);
	intersector->StartNewRendering();
	intersector->StartNewTraversal();
	pathCreateKernel<false> <<< dim3((w*h)/(32*8)+1,1,1), dim3(32, 8, 1)>>>(w, h, *intersector);
	int pass = 0;
	m_uNumRaysTraced = 0;
	do
	{
		m_uNumRaysTraced += intersector->IntersectBuffers<false>(0);
		unsigned int n = intersector[0][0].getCreatedRayCount();
		intersector->StartNewTraversal();
		pathIterateKernel<<< dim3(n/(32*8)+1,1,1), dim3(32, 8, 1)>>>(n, *I, pass, *intersector);
		if(pass != MAX_PASS)
			m_uNumRaysTraced += intersector->IntersectBuffers<true>(1);
	}
	while(intersector->getCreatedRayCount() && pass++ < MAX_PASS);
	m_uPassesDone++;
	I->DoUpdateDisplay(m_uPassesDone);
}

void k_FastTracer::DoRender(e_Image* I)
{
	doPath(I);
	//doDirect(I);
}