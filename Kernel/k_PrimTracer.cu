#include "k_PrimTracer.h"
#include "k_TraceHelper.h"
#include "k_TraceAlgorithms.h"
#include "..\Engine\e_Core.h"

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

CUDA_FUNC_IN Spectrum trace(Ray& r, CudaRNG& rng, float3* pout)
{
	TraceResult r2 = k_TraceRay(r);
	if(r2.hasHit())
	{
		if(pout)
			*pout = r(r2.m_fDist);
		BSDFSamplingRecord bRec;
		r2.getBsdfSample(r, rng, &bRec);
		//return Spectrum(clamp01(dot(bRec.map.sys.n, -r.direction)));
		Spectrum through(1.0f);
		if(g_SceneData.m_sVolume.HasVolumes())
			through = (-g_SceneData.m_sVolume.tau(r, 0, r2.m_fDist)).exp();
		Spectrum L = r2.Le(r(r2.m_fDist), bRec.map.sys, -r.direction);
		//return L + r2.getMat().bsdf.getDiffuseReflectance(bRec);
		float2 s = rng.randomFloat2();
		return L + r2.getMat().bsdf.sample(bRec, s) * through;
	}
	else return g_SceneData.EvalEnvironment(r);
}

CUDA_FUNC_IN Spectrum traceR(Ray& r, CudaRNG& rng, float3* pout)
{
	//return 0.5f + (rng.randomFloat() * 0.5f - 0.25f);

	const bool DIRECT = 1;
	TraceResult r2;
	r2.Init();
	Spectrum c = Spectrum(1.0f), L = Spectrum(0.0f);
	unsigned int depth = 0;
	bool specBounce = false;
	BSDFSamplingRecord bRec;
	while(k_TraceRay(r.direction, r.origin, &r2) && depth++ < 5)
	{
		if(pout && depth == 1)
			*pout = r(r2.m_fDist);
		if(g_SceneData.m_sVolume.HasVolumes())
			c = c * (-g_SceneData.m_sVolume.tau(r, 0, r2.m_fDist)).exp();
		r2.getBsdfSample(r, rng, &bRec);
		/*
		DirectSamplingRecord dRecLight(r(r2.m_fDist), bRec.ng, bRec.map.uv);
		Spectrum le = g_SceneData.sampleEmitterDirect(dRecLight, rng.randomFloat2());
		DirectSamplingRecord dRecSensor(r(r2.m_fDist), bRec.ng, bRec.map.uv);
		Spectrum im = g_SceneData.sampleSensorDirect(dRecSensor, rng.randomFloat2());
		if(!g_SceneData.Occluded(Ray(r(r2.m_fDist), dRecLight.d), 0, dRecLight.dist))
		{
			return im * 1000000.0f;

			float3 wi = normalize(dRecLight.p - r(r2.m_fDist));
			float3 wo = normalize(dRecSensor.p - r(r2.m_fDist));
			bRec.wi = bRec.map.sys.toLocal(wo);
			bRec.wo = bRec.map.sys.toLocal(wi);
			Spectrum f = r2.getMat().bsdf.f(bRec);
			float pdf = r2.getMat().bsdf.pdf(bRec);
			//return AbsDot(wi, bRec.map.sys.n) * AbsDot(wo, bRec.map.sys.n) * f / pdf * le;
			float pdf2 = 1.0f / (AbsDot(wo, bRec.map.sys.n) * AbsDot(wo, bRec.map.sys.n) * AbsDot(wo, bRec.map.sys.n)) * 1.0f / (dRecLight.dist * dRecLight.dist);
			return le * f / (pdf / pdf2);
		}
		else return 0.0f;*/
		//return Spectrum(dot(bRec.ng, -r.direction));
		if(depth == 1 || specBounce || !DIRECT)
			L += r2.Le(r(r2.m_fDist), bRec.map.sys, -r.direction);
		if(DIRECT)
			L += c * UniformSampleAllLights(bRec, r2.getMat(), 1);
		float pdf;
		Spectrum f = r2.getMat().bsdf.sample(bRec, pdf, rng.randomFloat2());

		float p = f.max();
		if (rng.randomFloat() < p)
			f = f / p;
		else break;

		c = c * f;
		if((bRec.sampledType & EDiffuse) == EDiffuse)
		{
			L += c;
			break;
		}
		specBounce = (bRec.sampledType & EDelta) != 0;
		r.origin = r(r2.m_fDist);
		r.direction = bRec.getOutgoing();
		r2.Init();
	}
	if(!r2.hasHit())
		L += c * g_SceneData.EvalEnvironment(r);
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
		float2 pixelSample = make_float2(x,y);
		
		Spectrum c = Spectrum(0.0f);
		float N2 = 1;
		for(float f = 0; f < N2; f++)
		{
			Ray r;
			Spectrum imp = g_SceneData.sampleSensorRay(r, pixelSample, rng.randomFloat2());
			float3 p = make_float3(0);
			c += imp * trace(r, rng, &p);
			if(fsumf(p))
			{
				uint3 pu = make_uint3(FloatToUInt(p.x), FloatToUInt(p.y), FloatToUInt(p.z));
				min3(&s_EyeHitBoxMin, pu);
				max3(&s_EyeHitBoxMax, pu);
			}
		}
		g_Image.AddSample(x, y, c / N2);
		
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
	Ray r = g_SceneData.GenerateSensorRay(p.x, p.y);
	CudaRNG rng = g_RNGData();
	Spectrum q =  trace(r, rng, 0);
}

void k_PrimTracer::DoRender(e_Image* I)
{
	ThrowCudaErrors();
	k_OnePassTracer::DoRender(I);
	unsigned int zero = 0;
	cudaMemcpyToSymbol(g_NextRayCounter2, &zero, sizeof(unsigned int));
	k_INITIALIZE(m_pScene, g_sRngs);
	uint3 ma = make_uint3(FloatToUInt(-FLT_MAX)), mi = make_uint3(FloatToUInt(FLT_MAX));
	cudaMemcpyToSymbol(g_EyeHitBoxMin2, &mi, 12);
	cudaMemcpyToSymbol(g_EyeHitBoxMax2, &ma, 12);
	primaryKernel<<< 180, dim3(32, MaxBlockHeight, 1)>>>(w, h, *I);
	cudaError_t r = cudaThreadSynchronize();
	k_TracerBase_update_TracedRays
	I->DoUpdateDisplay(1.0f);
	AABB m_sEyeBox;
	cudaMemcpyFromSymbol(&m_sEyeBox.minV, g_EyeHitBoxMin2, 12);
	cudaMemcpyFromSymbol(&m_sEyeBox.maxV, g_EyeHitBoxMax2, 12);
	m_sEyeBox.minV = make_float3(UIntToFloat(m_sEyeBox.minV.x), UIntToFloat(m_sEyeBox.minV.y), UIntToFloat(m_sEyeBox.minV.z));
	m_sEyeBox.maxV = make_float3(UIntToFloat(m_sEyeBox.maxV.x), UIntToFloat(m_sEyeBox.maxV.y), UIntToFloat(m_sEyeBox.maxV.z));
	m_pCamera->m_sLastFrustum = m_sEyeBox;
}

void k_PrimTracer::Debug(int2 pixel)
{
	//FW::printf("%f,%f",pixel.x/float(w),pixel.y/float(h));
	k_INITIALIZE(m_pScene, g_sRngs);
	//debugPixe2l<<<1,1>>>(w,h,pixel);
	Ray r = g_SceneData.GenerateSensorRay(pixel.x, pixel.y);
	CudaRNG rng = g_RNGData();
	trace(r, rng, 0);
}

void k_PrimTracer::CreateSliders(SliderCreateCallback a_Callback)
{
	//a_Callback(0,1,false,(float*)&m_bDirect,"%f Direct");
}