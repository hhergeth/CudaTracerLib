#include "k_PrimTracer.h"
#include "k_TraceHelper.h"
#include "k_IntegrateHelper.h"

CUDA_ONLY_FUNC float3 trace(Ray& r, CudaRNG& rng)
{
	MapParameters pm(float3(), make_float2(0.5f + atan2(r.direction.z, r.direction.x) * 0.5f * INV_PI, 0.5f - asin(r.direction.y) * INV_PI), Onb());
	TraceResult r2;
	r2.Init();
	float3 c = make_float3(1);
	unsigned int depth = 0;
	e_KernelBSDF bsdf;
	while(k_TraceRay<true>(r.direction, r.origin, &r2) && depth++ < 5)
	{
		if(g_SceneData.m_sVolume.HasVolumes())
			c = c * exp(-g_SceneData.m_sVolume.tau(r, 0, r2.m_fDist));
		float3 wi;
		float pdf;/*
		r2.GetBSDF(r(r2.m_fDist), g_SceneData.m_sMatData.Data, &bsdf);
		//((float3*)&bsdf.sys.m_tangent) = cross(bsdf.sys.m_binormal, bsdf.sys.m_normal);
		//return make_float3(dot(-r.direction, bsdf.sys.m_normal));
		BxDFType sampledType;
		float3 f = bsdf.Sample_f(-r.direction, &wi, BSDFSample(rng), &pdf, BSDF_ALL, &sampledType);
		f = f * AbsDot(wi, bsdf.sys.m_normal) / pdf;
		c = c * f;
		if((sampledType & BSDF_SPECULAR) != BSDF_SPECULAR)
			break;
		r.origin = r(r2.m_fDist);
		r.direction = wi;
		r2.Init();*/
	}
	if(r2.hasHit())
	{
		return c;// * UniformSampleAllLights(r(r2.m_fDist), bsdf.sys.m_normal, -r.direction, &bsdf, rng, 1);
	}
	else if(g_SceneData.m_sEnvMap.CanSample())
		c = c * g_SceneData.m_sEnvMap.Sample(r);
		//c = g_SceneData.m_sEnvMap.Map.Evaluate(pm);
	else c = make_float3(0);
	return c;
}

__global__ void primaryKernel(long long width, long long height, e_Image I)
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
		float N2 = 1;
		for(float f = 0; f < N2; f++)
		{
			CameraSample s = nextSample(x, y, rng);
			Ray r = g_CameraData.GenRay(s, width, height);
			c += trace(r, rng);
		}
		I.SetSampleDirect(nextSample(x, y, rng), c / N2);
		
		//Ray r = g_CameraData.GenRay(x, y, width, height, rng.randomFloat(), rng.randomFloat());
		//TraceResult r2 = k_TraceRay(r);
		//float3 c = make_float3(r2.m_fDist/length(g_SceneData.m_sBox.Size())*2.0f);
	}
	while(true);
	g_RNGData(rng);
}

__global__ void debugPixe2l(unsigned int width, unsigned int height, int2 p)
{
	Ray r = g_CameraData.GenRay(p, make_int2(width, height));
	//dir = make_float3(-0.98181188f, 0.18984018f, -0.0024534566f);
	//ori = make_float3(68790.375f, -12297.199f, 57510.383f);
	//ori += make_float3(g_SceneData.m_sTerrain.m_sMin.x, 0, g_SceneData.m_sTerrain.m_sMin.z);
	trace(r, g_RNGData());
}

static bool init = false;
void k_PrimTracer::DoRender(e_Image* I)
{
	if(!init)
	{
		init = true;
		cudaThreadSetLimit(cudaLimitStackSize, 2048);
	}
	m_sRngs.m_uOffset++;
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, m_sRngs);
	primaryKernel<<< 180, dim3(32, MaxBlockHeight, 1)>>>(w, h, *I);
	cudaError_t r = cudaThreadSynchronize();
	m_uRaysTraced = w * h;
	m_uPassesDone = 1;
	I->UpdateDisplay();
}

void k_PrimTracer::Debug(int2 pixel)
{
	m_pScene->UpdateInvalidated();
	e_KernelDynamicScene d2 = m_pScene->getKernelSceneData();
	k_INITIALIZE(d2);
	k_STARTPASS(m_pScene, m_pCamera, m_sRngs);
	debugPixe2l<<<1,1>>>(w,h,pixel);
}

void k_PrimTracer::CreateSliders(SliderCreateCallback a_Callback)
{
	
}