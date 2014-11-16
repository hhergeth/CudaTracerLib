#include "k_Tracer.h"
#include "k_TraceHelper.h"
#include "..\Engine\e_Core.h"
#include "k_TraceAlgorithms.h"

CUDA_DEVICE uint3 g_EyeHitBoxMin;
CUDA_DEVICE uint3 g_EyeHitBoxMax;
template<bool RECURSIVE> __global__ void k_GuessPass(int w, int h, float scx, float scy)
{
	int x = threadId % w, y = threadId / w;
	CudaRNG rng = g_RNGData();
	if(x < w && y < h)
	{
		Ray r = g_SceneData.GenerateSensorRay(float(x * scx), float(y * scy));
		TraceResult r2;
		r2.Init();
		int d = -1;
		while(k_TraceRay(r.direction, r.origin, &r2) && ++d < 5)
		{
			float3 p = r(r2.m_fDist);
			DifferentialGeometry dg;
			BSDFSamplingRecord bRec(dg);
			r2.getBsdfSample(r, rng, &bRec);
			r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
			r = Ray(dg.P, bRec.getOutgoing());
			float3 per = clamp01((p - g_SceneData.m_sBox.minV) / (g_SceneData.m_sBox.maxV - g_SceneData.m_sBox.minV)) * float(UINT_MAX);
			uint3 q = make_uint3(unsigned int(per.x), unsigned int(per.y), unsigned int(per.z));
			atomicMin(&g_EyeHitBoxMin.x, q.x);
			atomicMin(&g_EyeHitBoxMin.y, q.y);
			atomicMin(&g_EyeHitBoxMin.z, q.z);
			atomicMax(&g_EyeHitBoxMax.x, q.x);
			atomicMax(&g_EyeHitBoxMax.y, q.y);
			atomicMax(&g_EyeHitBoxMax.z, q.z);
			r2.Init();
			if(!RECURSIVE)
				break;
		}
	}
	g_RNGData(rng);
}

CUDA_FUNC_IN float3 lerp(const float3& a, const float3& b, const float3& t)
{
	return make_float3(lerp(a.x, b.x, t.x), lerp(a.y, b.y, t.y), lerp(a.z, b.z, t.z));
}

AABB k_Tracer::GetEyeHitPointBox(e_DynamicScene* m_pScene, e_Sensor* m_pCamera, bool recursive)
{
	ThrowCudaErrors();
	uint3 ma = make_uint3(0), mi = make_uint3(UINT_MAX);
	cudaMemcpyToSymbol(g_EyeHitBoxMin, &mi, 12);
	cudaMemcpyToSymbol(g_EyeHitBoxMax, &ma, 12);
	k_INITIALIZE(m_pScene, g_sRngs);
	int qw = 128, qh = 128, p0 = 16;
	float a = (float)m_pCamera->As()->m_resolution.x / qw, b = (float)m_pCamera->As()->m_resolution.y / qh;
	if(recursive)
		k_GuessPass<true> <<<dim3( qw/p0, qh/p0, 1), dim3(p0, p0, 1)>>>(qw, qh, a, b);
	else k_GuessPass<false> <<<dim3( qw/p0, qh/p0, 1), dim3(p0, p0, 1)>>>(qw, qh, a, b);
	cudaThreadSynchronize();
	ThrowCudaErrors();
	uint3 minU, maxU;
	cudaMemcpyFromSymbol(&minU, g_EyeHitBoxMin, 12);
	cudaMemcpyFromSymbol(&maxU, g_EyeHitBoxMax, 12);
	AABB m_sEyeBox;
	m_sEyeBox.minV = make_float3(float(minU.x), float(minU.y), float(minU.z)) / float(UINT_MAX);
	m_sEyeBox.maxV = make_float3(float(maxU.x), float(maxU.y), float(maxU.z)) / float(UINT_MAX);
	AABB box = m_pScene->getSceneBVH()->m_sBox;
	m_sEyeBox.maxV = lerp(box.minV, box.maxV, m_sEyeBox.maxV);
	m_sEyeBox.minV = lerp(box.minV, box.maxV, m_sEyeBox.minV);
	return m_sEyeBox;
}

CUDA_DEVICE TraceResult res;
CUDA_GLOBAL void traceKernel(Ray r)
{
	res.Init();
	res = k_TraceRay(r);
}

TraceResult k_Tracer::TraceSingleRay(Ray r, e_DynamicScene* s, e_Sensor* c)
{
	CudaRNGBuffer tmp;
	k_INITIALIZE(s, tmp);
	//traceKernel<<<1,1>>>(r);
	return k_TraceRay(r);
	//TraceResult r2;
	//cudaMemcpyFromSymbol(&r2, res, sizeof(r2));
	//return r2;
}

CUDA_DEVICE unsigned int g_ShotRays;
CUDA_DEVICE unsigned int g_SuccRays;

__global__ void estimateLightVisibility(int w, int h, float scx, float scy, int recursion_depth)
{
	int x = threadId % w, y = threadId / w;
	CudaRNG rng = g_RNGData();
	if (x < w && y < h)
	{
		Ray r = g_SceneData.GenerateSensorRay(float(x * scx), float(y * scy));
		TraceResult r2;
		r2.Init();
		int d = -1;
		unsigned int N = 0, S = 0;
		while (k_TraceRay(r.direction, r.origin, &r2) && ++d < recursion_depth)
		{
			DifferentialGeometry dg;
			BSDFSamplingRecord bRec(dg);
			r2.getBsdfSample(r, rng, &bRec);

			for (int i = 0; i < g_SceneData.m_sLightData.Length; i++)
			{
				PositionSamplingRecord pRec;
				g_SceneData.m_sLightData[i].samplePosition(pRec, rng.randomFloat2());
				bool v = V(pRec.p, dg.P);
				N++;
				S += v;
			}

			r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
			r = Ray(dg.P, bRec.getOutgoing());
		}
		atomicAdd(&g_ShotRays, N);
		atomicAdd(&g_SuccRays, S);
	}
}

float k_Tracer::GetLightVisibility(e_DynamicScene* s, e_Sensor* c, int recursion_depth)
{
	unsigned int zero = 0;
	cudaMemcpyToSymbol(g_ShotRays, &zero, sizeof(unsigned int));
	cudaMemcpyToSymbol(g_SuccRays, &zero, sizeof(unsigned int));
	k_INITIALIZE(s, g_sRngs);
	int qw = 128, qh = 128, p0 = 16;
	float a = (float)c->As()->m_resolution.x / qw, b = (float)c->As()->m_resolution.y / qh;
	estimateLightVisibility << <dim3(qw / p0, qh / p0, 1), dim3(p0, p0, 1) >> >(qw, qh, a, b, recursion_depth);
	cudaThreadSynchronize();
	unsigned int N, S;
	cudaMemcpyFromSymbol(&N, g_ShotRays, sizeof(unsigned int));
	cudaMemcpyFromSymbol(&S, g_SuccRays, sizeof(unsigned int));
	return float(S) / float(N);
}