#include "k_Tracer.h"
#include "k_TraceHelper.h"

//SHOULD NOT WORK
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

/*
CUDA_FUNC_IN unsigned int FloatToUInt(float f)
{
	unsigned int mask = -unsigned int(*(unsigned int*)&f >> 31) | 0x80000000;
	return (*(unsigned int*)&f) ^ mask;
}

CUDA_FUNC_IN float UIntToFloat(unsigned int f)
{
	unsigned int mask = ((f >> 31) - 1) | 0x80000000, q = f ^ mask;
	return *(float*)&q;
}
*/

CUDA_DEVICE uint3 g_EyeHitBoxMin;
CUDA_DEVICE uint3 g_EyeHitBoxMax;
__global__ void k_GuessPass(int w, int h)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y, N = y * w + x;
	CudaRNG localState = g_RNGData();
	if(x < w && y < h)
	{
		CameraSample s = nextSample(x, y, localState);
		Ray r = g_CameraData.GenRay(s, w, h);
		TraceResult r2;
		r2.Init();
		int d = -1;
		while(k_TraceRay(r.direction, r.origin, &r2) && ++d < 10)
		{
			float3 p = r(r2.m_fDist);
			e_KernelBSDF bsdf = r2.GetBSDF(p);
			float3 inc;
			float pdf;
			float3 col = bsdf.Sample_f(-1.0f * r.direction, &inc, BSDFSample(localState), &pdf);
			r = Ray(r(r2.m_fDist), inc);
			uint3 pu = make_uint3(FloatToUInt(p.x), FloatToUInt(p.y), FloatToUInt(p.z));
			atomicMin(&g_EyeHitBoxMin.x, pu.x);
			atomicMin(&g_EyeHitBoxMin.y, pu.y);
			atomicMin(&g_EyeHitBoxMin.z, pu.z);
			atomicMax(&g_EyeHitBoxMax.x, pu.x);
			atomicMax(&g_EyeHitBoxMax.y, pu.y);
			atomicMax(&g_EyeHitBoxMax.z, pu.z);
			r2.Init();
		}
	}
	g_RNGData(localState);
}

AABB k_Tracer::GetEyeHitPointBox(e_DynamicScene* m_pScene, e_Camera* m_pCamera)
{
	uint3 ma = make_uint3(FloatToUInt(-FLT_MAX)), mi = make_uint3(FloatToUInt(FLT_MAX));
	cudaMemcpyToSymbol(g_EyeHitBoxMin, &mi, 12);
	cudaMemcpyToSymbol(g_EyeHitBoxMax, &ma, 12);
	k_INITIALIZE(m_pScene->getKernelSceneData());
	k_STARTPASS(m_pScene, m_pCamera, g_sRngs);
	int qw = 128, qh = 128, p0 = 16;
	k_GuessPass<<<dim3( qw/p0, qh/p0, 1), dim3(p0, p0, 1)>>>(qw, qh);
	cudaThreadSynchronize();
	AABB m_sEyeBox;
	cudaMemcpyFromSymbol(&m_sEyeBox.minV, g_EyeHitBoxMin, 12);
	cudaMemcpyFromSymbol(&m_sEyeBox.maxV, g_EyeHitBoxMax, 12);
	m_sEyeBox.minV = make_float3(UIntToFloat(m_sEyeBox.minV.x), UIntToFloat(m_sEyeBox.minV.y), UIntToFloat(m_sEyeBox.minV.z));
	m_sEyeBox.maxV = make_float3(UIntToFloat(m_sEyeBox.maxV.x), UIntToFloat(m_sEyeBox.maxV.y), UIntToFloat(m_sEyeBox.maxV.z));
	return m_sEyeBox;
}

CUDA_DEVICE TraceResult res;
CUDA_GLOBAL void traceKernel(Ray r)
{
	res.Init();
	res = k_TraceRay(r);
}

TraceResult k_Tracer::TraceSingleRay(Ray r, e_DynamicScene* s, e_Camera* c)
{
	k_TracerRNGBuffer tmp;
	s->UpdateInvalidated();
	k_INITIALIZE(s->getKernelSceneData());
	k_STARTPASS(s, c, tmp);
	traceKernel<<<1,1>>>(r);
	TraceResult r2;
	cudaMemcpyFromSymbol(&r2, res, sizeof(r2));
	return r2;
}