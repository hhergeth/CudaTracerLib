#include "k_PathTracer.h"
#include "..\Kernel\k_TraceHelper.h"
#include <time.h>
#include "..\Kernel\k_TraceAlgorithms.h"

CUDA_ALIGN(16) CUDA_DEVICE unsigned int g_NextRayCounter;

template<bool DIRECT> CUDA_FUNC_IN Spectrum PathTraceTTT(float3& a_Dir, float3& a_Ori, CudaRNG& rnd, float* distTravalled = 0)
{
	Ray r0 = Ray(a_Ori, a_Dir);
	TraceResult r;
	r.Init();
	Spectrum cl = Spectrum(0.0f);   // accumulated color
	Spectrum cf = Spectrum(1.0f);  // accumulated reflectance
	int depth = 0;
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	while (k_TraceRay(r0.direction, r0.origin, &r) && depth++ < 7)
	{
		r.getBsdfSample(r0, rnd, &bRec); //return (Spectrum(bRec.map.sys.n) + Spectrum(1)) / 2.0f; //return bRec.map.sys.n;
		cl += cf * r.Le(r0(r.m_fDist), bRec.dg.sys, -r0.direction);
		Spectrum f;
		if (DIRECT)
		{
			DirectSamplingRecord dRec(bRec.dg.P, bRec.dg.sys.n);
			Spectrum value = g_SceneData.m_sLightData[0].sampleDirect(dRec, rnd.randomFloat2());
			bRec.wo = normalize(bRec.dg.toLocal(dRec.d));
			f = r.getMat().bsdf.f(bRec) / g_SceneData.m_sLightData[0].As<e_DiffuseLight>()->shapeSet.Pdf(dRec) * G(bRec.dg.sys.n, dRec.n, bRec.dg.P, dRec.p);
		}
		else f = r.getMat().bsdf.sample(bRec, rnd.randomFloat2());

		cf = cf * f;
		r0 = Ray(r0(r.m_fDist), bRec.getOutgoing());
		r.Init();
	}
	return cl;
}

template<bool DIRECT> CUDA_FUNC_IN Spectrum PathTrace(Ray& r, const Ray& rX, const Ray& rY, CudaRNG& rnd, float* distTravalled = 0)
{
	TraceResult r2;
	r2.Init();
	Spectrum cl = Spectrum(0.0f);   // accumulated color
	Spectrum cf = Spectrum(1.0f);  // accumulated reflectance
	int depth = 0;
	bool specularBounce = false;
	DifferentialGeometry dg;
	BSDFSamplingRecord bRec(dg);
	while (k_TraceRay(r.direction, r.origin, &r2) && depth++ < 7)
	{
		if (distTravalled && depth == 1)
			*distTravalled = r2.m_fDist;
		r2.getBsdfSample(r, rnd, &bRec);// return (Spectrum(bRec.map.sys.n) + Spectrum(1)) / 2.0f; //return bRec.map.sys.n;
		if (depth == 1)
			dg.computePartials(r, rX, rY);
		if (!DIRECT || (depth == 1 || specularBounce))
			cl += cf * r2.Le(bRec.dg.P, bRec.dg.sys, -r.direction);
		Spectrum f = r2.getMat().bsdf.sample(bRec, rnd.randomFloat2());
		if (DIRECT)
			cl += cf * UniformSampleAllLights(bRec, r2.getMat(), 1);
		specularBounce = (bRec.sampledType & EDelta) != 0;
		float p = f.max();
		if (depth > 5)
		if (rnd.randomFloat() < p)
			f = f / p;
		else break;
		if (f.isZero())
			break;
		cf = cf * f;
		r = Ray(dg.P, bRec.getOutgoing());
		r2.Init();
	}
	if (!r2.hasHit() && depth == 0)
		cl = cf * g_SceneData.EvalEnvironment(r, rX, rY);
	else cl += cf * g_SceneData.EvalEnvironment(r);
	return cl;
}

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

		float2 screenPos = make_float2(rayidx % width, rayidx / width) + rng.randomFloat2();
		Ray r, rX, rY;
		Spectrum imp = g_SceneData.sampleSensorRay(r, rX, rY, screenPos, rng.randomFloat2());

		Spectrum col = imp * PathTrace<DIRECT>(r, rX, rY, rng);
		
		g_Image.AddSample(screenPos.x, screenPos.y, col);
	}
	while(true);
	g_RNGData(rng);
}

__global__ void debugPixel(unsigned int width, unsigned int height, int2 p)
{
	CudaRNG rng = g_RNGData();
	Ray r = g_SceneData.GenerateSensorRay(p.x, p.y);	
	PathTrace<true>(r, r, r, rng);
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
	PathTrace<true>(r, r, r, rng);
}

template<bool DIRECT> __global__ void pathKernel2(unsigned int width, unsigned int height, e_Image g_Image, k_BlockSampler sampler)
{
	uint2 pixel = sampler.pixelCoord();
	CudaRNG rng = g_RNGData();
	if(pixel.x < width && pixel.y < height)
	{
		Ray r;
		Spectrum imp = g_SceneData.sampleSensorRay(r, make_float2(pixel.x, pixel.y), rng.randomFloat2());
		Spectrum col = imp * PathTrace<DIRECT>(r, r, r, rng);
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