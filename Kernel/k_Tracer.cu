#include "k_Tracer.h"
#include "k_TraceHelper.h"
#include "k_TraceAlgorithms.h"
#include "../Engine/e_DynamicScene.h"
#include "../Engine/e_Light.h"

CUDA_DEVICE uint3 g_EyeHitBoxMin;
CUDA_DEVICE uint3 g_EyeHitBoxMax;
template<bool RECURSIVE> __global__ void k_GuessPass(int w, int h, float scx, float scy)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	CudaRNG rng = g_RNGData();
	if(x < w && y < h)
	{
		Ray r = g_SceneData.GenerateSensorRay(float(x * scx), float(y * scy));
		TraceResult r2;
		r2.Init();
		int d = -1;
		while(k_TraceRay(r.direction, r.origin, &r2) && ++d < 5)
		{
			DifferentialGeometry dg;
			BSDFSamplingRecord bRec(dg);
			r2.getBsdfSample(r, bRec, ETransportMode::ERadiance, &rng);
			r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
			r = Ray(dg.P, bRec.getOutgoing());
			Vec3f per = math::clamp01((dg.P - g_SceneData.m_sBox.minV) / (g_SceneData.m_sBox.maxV - g_SceneData.m_sBox.minV)) * float(UINT_MAX);
			Vec3u q = Vec3u(unsigned int(per.x), unsigned int(per.y), unsigned int(per.z));
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

AABB k_TracerBase::GetEyeHitPointBox(e_DynamicScene* m_pScene, bool recursive)
{
	ThrowCudaErrors();
	Vec3u ma = Vec3u(0), mi = Vec3u(UINT_MAX);
	ThrowCudaErrors(cudaMemcpyToSymbol(g_EyeHitBoxMin, &mi, 12));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_EyeHitBoxMax, &ma, 12));
	k_INITIALIZE(m_pScene, g_sRngs);
	int qw = 128, qh = 128, p0 = 16;
	float a = (float)m_pScene->getCamera()->As()->m_resolution.x / qw, b = (float)m_pScene->getCamera()->As()->m_resolution.y / qh;
	if(recursive)
		k_GuessPass<true> <<<dim3( qw/p0, qh/p0, 1), dim3(p0, p0, 1)>>>(qw, qh, a, b);
	else k_GuessPass<false> <<<dim3( qw/p0, qh/p0, 1), dim3(p0, p0, 1)>>>(qw, qh, a, b);
	ThrowCudaErrors(cudaDeviceSynchronize());
	uint3 minU, maxU;
	ThrowCudaErrors(cudaMemcpyFromSymbol(&minU, g_EyeHitBoxMin, 12));
	ThrowCudaErrors(cudaMemcpyFromSymbol(&maxU, g_EyeHitBoxMax, 12));
	AABB m_sEyeBox;
	m_sEyeBox.minV = Vec3f(float(minU.x), float(minU.y), float(minU.z)) / float(UINT_MAX);
	m_sEyeBox.maxV = Vec3f(float(maxU.x), float(maxU.y), float(maxU.z)) / float(UINT_MAX);
	AABB box = g_SceneData.m_sBox;
	m_sEyeBox.maxV = math::lerp(box.minV, box.maxV, m_sEyeBox.maxV);
	m_sEyeBox.minV = math::lerp(box.minV, box.maxV, m_sEyeBox.minV);
	return m_sEyeBox;
}

CUDA_DEVICE TraceResult res;
CUDA_GLOBAL void traceKernel(Ray r)
{
	res.Init();
	res = k_TraceRay(r);
}

TraceResult k_TracerBase::TraceSingleRay(Ray r, e_DynamicScene* s)
{
	CudaRNGBuffer tmp;
	k_INITIALIZE(s, tmp);
	return k_TraceRay(r);
}

CUDA_DEVICE unsigned int g_ShotRays;
CUDA_DEVICE unsigned int g_SuccRays;

__global__ void estimateLightVisibility(int w, int h, float scx, float scy, int recursion_depth)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
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
			r2.getBsdfSample(r, bRec, ETransportMode::ERadiance, &rng);

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

float k_TracerBase::GetLightVisibility(e_DynamicScene* s, int recursion_depth)
{
	unsigned int zero = 0;
	ThrowCudaErrors(cudaMemcpyToSymbol(g_ShotRays, &zero, sizeof(unsigned int)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_SuccRays, &zero, sizeof(unsigned int)));
	k_INITIALIZE(s, g_sRngs);
	int qw = 128, qh = 128, p0 = 16;
	float a = (float)s->getCamera()->As()->m_resolution.x / qw, b = (float)s->getCamera()->As()->m_resolution.y / qh;
	estimateLightVisibility << <dim3(qw / p0, qh / p0, 1), dim3(p0, p0, 1) >> >(qw, qh, a, b, recursion_depth);
	ThrowCudaErrors(cudaDeviceSynchronize());
	unsigned int N, S;
	ThrowCudaErrors(cudaMemcpyFromSymbol(&N, g_ShotRays, sizeof(unsigned int)));
	ThrowCudaErrors(cudaMemcpyFromSymbol(&S, g_SuccRays, sizeof(unsigned int)));
	return float(S) / float(N);
}

__global__ void depthKernel(e_Image I)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x < I.getWidth() && y < I.getHeight())
	{
		Ray r = g_SceneData.GenerateSensorRay(x, y);
		TraceResult r2 = k_TraceRay(r);
		float d = 1.0f;
		if (r2.hasHit())
			d = CalcZBufferDepth(g_SceneData.m_Camera.As()->m_fNearFarDepths.x, g_SceneData.m_Camera.As()->m_fNearFarDepths.y, r2.m_fDist);
		I.SetSample(x, y, *(RGBCOL*)&d);
	}
}

void k_TracerBase::RenderDepth(e_Image* img, e_DynamicScene* s)
{
	k_INITIALIZE(s, g_sRngs);
	depthKernel << <dim3(img->getWidth() / 16 + 1, img->getHeight() / 16 + 1), dim3(16, 16) >> >(*img);
	ThrowCudaErrors(cudaDeviceSynchronize());
}