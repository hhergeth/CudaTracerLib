#include "k_FastTracer.h"
#include "..\Kernel\k_TraceHelper.h"
#include "..\Kernel\k_TraceAlgorithms.h"

__global__ void pathCreateKernel(unsigned int w, unsigned int h, k_PTDBuffer g_Intersector)
{
	int idx = threadId;
	if(idx >= w * h)
		return;
	int x = idx % w, y = idx / w;
	Ray r;
	Spectrum W = g_SceneData.sampleSensorRay(r, Vec2f(x,y), Vec2f(0,0));
	traversalRay& ray = g_Intersector(idx, 0);
	ray.a = Vec4f(r.origin, 0.0f);
	ray.b = Vec4f(r.direction, FLT_MAX);
	rayData& dat = g_Intersector(idx);
	dat.x = x;
	dat.y = y;
	dat.throughput = W;
	dat.L = Spectrum(0.0f);
	dat.dIdx = 0xffffffff;
}

__global__ void doDirectKernel(unsigned int w, unsigned int h, k_PTDBuffer g_Intersector, e_Image I, float SCALE)
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

		rayData r = a_RayBuffer[rayidx];
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
	unsigned int idx = threadId;
	if(idx >= w * h)
		return;
	traversalResult& res = g_Intersector.res(idx, 0);
	RGBCOL col;
	col.x = col.w = 255;
	col.y = col.z = 0;
	if(res.dist)
	{
		//tar[rayidx] = Spectrum(a_ResBuffer[rayidx].m_fDist/SCALE).toRGBCOL();
		float f = res.dist/SCALE * 255.0f;
		unsigned char c = (unsigned char)f;
		unsigned int i = (255 << 24) | (c << 16) | (c << 8) | c;
		col = *(RGBCOL*)&i;
	}
	Spectrum s;
	s.fromRGBCOL(col);
	I.AddSample(idx % w, idx / w, s);
}

__device__ CUDA_INLINE Vec2f stratifiedSample(const Vec2f& f, int pass)
{
	return f;
	//int i = pass % 64;
	//int x = i % 8, y = i / 8;
	//return Vec2f(x, y) / 8.0f + f / 8.0f;
}

#define max_PASS 5
CUDA_DEVICE k_PTDBuffer g_Intersector;
CUDA_DEVICE k_PTDBuffer g_Intersector2;
CUDA_DEVICE int g_NextRayCounterFT;
__global__ void pathIterateKernel(unsigned int N, e_Image I, int pass, int iterationIdx)//template
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
		if (pass > 0 && dat.dIdx != 0xffffffff)
		{
			traversalResult& res = g_Intersector.res(dat.dIdx, 1);
			traversalRay& ray = g_Intersector(dat.dIdx, 1);
			if (res.dist >= dat.dDist * 0.95f)
				dat.L += dat.throughput * dat.directF;
		}

		traversalResult& res = g_Intersector.res(rayidx, 0);
		traversalRay& ray = g_Intersector(rayidx, 0);

		if (res.dist)
		{
			Ray r(ray.a.getXYZ(), ray.b.getXYZ());
			TraceResult r2;
			res.toResult(&r2, g_SceneData);
			DifferentialGeometry dg;
			BSDFSamplingRecord bRec(dg);
			r2.getBsdfSample(r, rng, &bRec);
			if (pass == 0 || dat.dIdx == 0xffffffff)
				dat.L += r2.Le(bRec.dg.P, bRec.dg.sys, -r.direction) * dat.throughput;
			if (pass + 1 != max_PASS)
			{
				Spectrum f = r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
				dat.throughput *= f;
				unsigned int idx2 = g_Intersector2.insertRay(0);
				traversalRay& ray2 = g_Intersector2(idx2, 0);
				ray2.a = Vec4f(bRec.dg.P, 1e-2f);
				ray2.b = Vec4f(bRec.getOutgoing(), FLT_MAX);

				DirectSamplingRecord dRec(bRec.dg.P, bRec.dg.sys.n);
				Spectrum value = g_SceneData.sampleEmitterDirect(dRec, rng.randomFloat2());
				if (r2.getMat().bsdf.hasComponent(ESmooth) && !value.isZero())
				{
					bRec.wo = normalize(bRec.dg.toLocal(dRec.d));
					Spectrum bsdfVal = r2.getMat().bsdf.f(bRec);
					const float bsdfPdf = r2.getMat().bsdf.pdf(bRec);
					const float weight = MonteCarlo::PowerHeuristic(1, dRec.pdf, 1, bsdfPdf);
					dat.directF = value * bsdfVal * weight;
					dat.dDist = dRec.dist;
					dat.dIdx = g_Intersector2.insertRay(1);
					traversalRay& ray3 = g_Intersector2(dat.dIdx, 1);
					ray3.a = Vec4f(bRec.dg.P, 1e-2f);
					ray3.b = Vec4f(dRec.d, FLT_MAX);
				}
				else dat.dIdx = 0xffffffff;
				g_Intersector2(idx2) = dat;
			}
		}
		else dat.L += dat.throughput * g_SceneData.EvalEnvironment(Ray(ray.a.getXYZ(), ray.b.getXYZ()));

		if (!res.dist || pass + 1 == max_PASS)
			I.AddSample(dat.x, dat.y, dat.L);
	} while (true);
	g_RNGData(rng);
}

#include "..\Base\Timer.h"
static cTimer TT;
void k_FastTracer::doDirect(e_Image* I)
{
	k_PTDBuffer* buf = bufA;
	float scl = length(g_SceneData.m_sBox.Size());
	pathCreateKernel<<< dim3((w*h)/(32*8)+1,1,1), dim3(32, 8, 1)>>>(w, h, *buf);
	buf->setNumRays(w * h, 0);
	
	buf->IntersectBuffers<false>(m_pScene->getNodeCount() == 1);

	I->Clear();
	doDirectKernel<<< dim3((w*h)/(32*8)+1,1,1), dim3(32, 8, 1)>>>(w, h, *buf, *I, scl);
}

void k_FastTracer::doPath(e_Image* I)
{
	bufA->Clear();
	pathCreateKernel<<< dim3((w*h)/(32*8)+1,1,1), dim3(32, 8, 1)>>>(w, h, *bufA);
	bufA->setNumRays(w * h, 0);
	int pass = 0, zero = 0;
	k_PTDBuffer* srcBuf = bufA, *destBuf = bufB;
	do
	{
		destBuf->Clear();
		srcBuf->IntersectBuffers<false>(m_pScene->getNodeCount() == 1);
		cudaMemcpyToSymbol(g_Intersector, srcBuf, sizeof(*srcBuf));
		cudaMemcpyToSymbol(g_Intersector2, destBuf, sizeof(*destBuf));
		cudaMemcpyToSymbol(g_NextRayCounterFT, &zero, sizeof(zero));
		pathIterateKernel << < dim3(180, 1, 1), dim3(32, 6, 1) >> >(srcBuf->getNumRays(0), *I, pass, m_uPassesDone);
		cudaMemcpyFromSymbol(srcBuf, g_Intersector, sizeof(*srcBuf));
		cudaMemcpyFromSymbol(destBuf, g_Intersector2, sizeof(*destBuf));
		swapk(srcBuf, destBuf);
	}
	while (srcBuf->getNumRays(0) && ++pass < max_PASS);
}

void k_FastTracer::DoRender(e_Image* I)
{
	doPath(I);
	//doDirect(I);
}