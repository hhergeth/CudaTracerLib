#include "k_PrimTracer.h"
#include "k_TraceHelper.h"

CUDA_ALIGN(16) CUDA_DEVICE unsigned int g_NextRayCounter2;
CUDA_DEVICE uint3 g_EyeHitBoxMin2;
CUDA_DEVICE uint3 g_EyeHitBoxMax2;

CUDA_FUNC_IN unsigned int FloatToUInt(float f2)
{
	unsigned int f = *(unsigned int*)&f2;
	unsigned int mask = -int(f >> 31) | 0x80000000;
	return f ^ mask;
}

CUDA_FUNC_IN float UIntToFloat(float f2)
{
	unsigned int f = *(unsigned int*)&f2;
	unsigned int mask = ((f >> 31) - 1) | 0x80000000;
	unsigned int i = f ^ mask;
	return *(float*)&i;
}

CUDA_ONLY_FUNC void min3(uint3* tar, uint3& val)
{
	atomicMin(&tar->x, val.x);
	atomicMin(&tar->y, val.y);
	atomicMin(&tar->z, val.z);
}

CUDA_ONLY_FUNC void max3(uint3* tar, uint3& val)
{
	atomicMax(&tar->x, val.x);
	atomicMax(&tar->y, val.y);
	atomicMax(&tar->z, val.z);
}

CUDA_ONLY_FUNC float3 trace(Ray& r, CudaRNG& rng, float3* pout)
{
	const bool DIRECT = false;
	TraceResult r2;
	r2.Init();
	float3 c = make_float3(1), L = make_float3(0);
	unsigned int depth = 0;
	e_KernelBSDF bsdf;
	bool specBounce = false;
	while(k_TraceRay(r.direction, r.origin, &r2) && depth++ < 5)
	{
		if(pout && depth == 1)
			*pout = r(r2.m_fDist);
		if(g_SceneData.m_sVolume.HasVolumes())
			c = c * exp(-g_SceneData.m_sVolume.tau(r, 0, r2.m_fDist));
		float3 wi;
		float pdf;
		r2.GetBSDF(r(r2.m_fDist), &bsdf);
		if(depth == 1 || specBounce || !DIRECT)
			L += r2.Le(r(r2.m_fDist), bsdf.sys.n, -r.direction);
		//if(DIRECT)
		//	L += c * UniformSampleAllLights(r(r2.m_fDist), bsdf.sys.n, -r.direction, &bsdf, rng, 1);
		//((float3*)&bsdf.sys.t) = cross(bsdf.sys.s, bsdf.sys.n);
		//return make_float3(dot(-r.direction, bsdf.sys.n));
		BxDFType sampledType;
		float3 f = bsdf.Sample_f(-r.direction, &wi, BSDFSample(rng), &pdf, BSDF_ALL, &sampledType);
		f = f * AbsDot(wi, bsdf.sys.n) / pdf;
		c = c * f;
		if((sampledType & BSDF_DIFFUSE) == BSDF_DIFFUSE)
		{
			L += c;
			break;
		}
		specBounce = (sampledType & BSDF_SPECULAR) != 0;
		r.origin = r(r2.m_fDist);
		r.direction = wi;
		r2.Init();
	}
	if(!r2.hasHit() && g_SceneData.m_sEnvMap.CanSample())
		L += c * g_SceneData.m_sEnvMap.Sample(r);
	else if(!r2.hasHit()) 
		L = make_float3(0);
	return L;
}

CUDA_SHARED uint3 s_EyeHitBoxMin;
CUDA_SHARED uint3 s_EyeHitBoxMax;
__global__ void primaryKernel(long long width, long long height, e_Image g_Image)
{
	if(!threadIdx.x && !threadIdx.y)
	{
		s_EyeHitBoxMax = make_uint3(FloatToUInt(-FLT_MAX));
		s_EyeHitBoxMin = make_uint3(FloatToUInt(+FLT_MAX));
	}
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
				rayBase = atomicAdd(&g_NextRayCounter2, numTerminated);

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
			float3 p = make_float3(0);
			c += trace(r, rng, &p);
			if(fsumf(p))
			{
				uint3 pu = make_uint3(FloatToUInt(p.x), FloatToUInt(p.y), FloatToUInt(p.z));
				min3(&s_EyeHitBoxMin, pu);
				max3(&s_EyeHitBoxMax, pu);
			}
		}
		g_Image.SetSampleDirect(nextSample(x, y, rng), c / N2);
		
		//Ray r = g_CameraData.GenRay(x, y, width, height, rng.randomFloat(), rng.randomFloat());
		//TraceResult r2 = k_TraceRay(r);
		//float3 c = make_float3(r2.m_fDist/length(g_SceneData.m_sBox.Size())*2.0f);
	}
	while(true);
	g_RNGData(rng);
	if(!threadIdx.x && !threadIdx.y)
	{
		min3(&g_EyeHitBoxMin2, s_EyeHitBoxMin);
		max3(&g_EyeHitBoxMax2, s_EyeHitBoxMax);
	}
}

__global__ void debugPixe2l(unsigned int width, unsigned int height, int2 p)
{
	Ray r = g_CameraData.GenRay(p, make_int2(width, height));
	//dir = make_float3(-0.98181188f, 0.18984018f, -0.0024534566f);
	//ori = make_float3(68790.375f, -12297.199f, 57510.383f);
	//ori += make_float3(g_SceneData.m_sTerrain.m_sMin.x, 0, g_SceneData.m_sTerrain.m_sMin.z);
	trace(r, g_RNGData(), 0);
}

void k_PrimTracer::DoRender(e_Image* I)
{
	k_OnePassTracer::DoRender(I);
	unsigned int zero = 0;
	cudaMemcpyToSymbol(g_NextRayCounter2, &zero, sizeof(unsigned int));
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, g_sRngs);
	uint3 ma = make_uint3(FloatToUInt(-FLT_MAX)), mi = make_uint3(FloatToUInt(FLT_MAX));
	cudaMemcpyToSymbol(g_EyeHitBoxMin2, &mi, 12);
	cudaMemcpyToSymbol(g_EyeHitBoxMax2, &ma, 12);
	primaryKernel<<< 180, dim3(32, MaxBlockHeight, 1)>>>(w, h, *I);
	cudaError_t r = cudaThreadSynchronize();
	k_TracerBase_update_TracedRays
	I->UpdateDisplay();
	AABB m_sEyeBox;
	cudaMemcpyFromSymbol(&m_sEyeBox.minV, g_EyeHitBoxMin2, 12);
	cudaMemcpyFromSymbol(&m_sEyeBox.maxV, g_EyeHitBoxMax2, 12);
	m_sEyeBox.minV = make_float3(UIntToFloat(m_sEyeBox.minV.x), UIntToFloat(m_sEyeBox.minV.y), UIntToFloat(m_sEyeBox.minV.z));
	m_sEyeBox.maxV = make_float3(UIntToFloat(m_sEyeBox.maxV.x), UIntToFloat(m_sEyeBox.maxV.y), UIntToFloat(m_sEyeBox.maxV.z));
	m_pCamera->m_sLastFrustum = m_sEyeBox;
}

void k_PrimTracer::Debug(int2 pixel)
{
	m_pScene->UpdateInvalidated();
	e_KernelDynamicScene d2 = m_pScene->getKernelSceneData();
	k_INITIALIZE(d2);
	k_STARTPASS(m_pScene, m_pCamera, g_sRngs);
	debugPixe2l<<<1,1>>>(w,h,pixel);
}

void k_PrimTracer::CreateSliders(SliderCreateCallback a_Callback)
{
	//a_Callback(0,1,false,(float*)&m_bDirect,"%f Direct");
}