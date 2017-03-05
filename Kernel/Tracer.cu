#include "Tracer.h"
#include "TraceHelper.h"
#include "TraceAlgorithms.h"
#include <Engine/DynamicScene.h>
#include <Engine/Light.h>
#include "Sampler.h"

namespace CudaTracerLib {

CUDA_DEVICE uint3 g_EyeHitBoxMin;
CUDA_DEVICE uint3 g_EyeHitBoxMax;
template<bool RECURSIVE> __global__ void k_GuessPass(int w, int h, float scx, float scy)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	auto rng = g_SamplerData(y * w + x);
	if (x < w && y < h)
	{
		NormalizedT<Ray> r = g_SceneData.GenerateSensorRay(float(x * scx), float(y * scy));
		TraceResult r2;
		r2.Init();
		int d = -1;
		while (traceRay(r.dir(), r.ori(), &r2) && ++d < 5)
		{
			DifferentialGeometry dg;
			BSDFSamplingRecord bRec(dg);
			r2.getBsdfSample(r, bRec, ETransportMode::ERadiance);
			r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
			r = NormalizedT<Ray>(dg.P, bRec.getOutgoing());
			Vec3f per = math::clamp01((dg.P - g_SceneData.m_sBox.minV) / (g_SceneData.m_sBox.maxV - g_SceneData.m_sBox.minV)) * float(UINT_MAX);
			Vec3u q = Vec3u((unsigned int)per.x, (unsigned int)per.y, (unsigned int)per.z);
			atomicMin(&g_EyeHitBoxMin.x, q.x);
			atomicMin(&g_EyeHitBoxMin.y, q.y);
			atomicMin(&g_EyeHitBoxMin.z, q.z);
			atomicMax(&g_EyeHitBoxMax.x, q.x);
			atomicMax(&g_EyeHitBoxMax.y, q.y);
			atomicMax(&g_EyeHitBoxMax.z, q.z);
			r2.Init();
			if (!RECURSIVE)
				break;
		}
	}
}

AABB TracerBase::GetEyeHitPointBox(DynamicScene* m_pScene, bool recursive)
{
	Vec3u ma = Vec3u(0), mi = Vec3u(UINT_MAX);
	ThrowCudaErrors(cudaMemcpyToSymbol(g_EyeHitBoxMin, &mi, 12));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_EyeHitBoxMax, &ma, 12));
	UpdateKernel(m_pScene);
	int qw = 128, qh = 128, p0 = 16;
	float a = (float)m_pScene->getCamera()->As()->m_resolution.x / qw, b = (float)m_pScene->getCamera()->As()->m_resolution.y / qh;
	if (recursive)
		k_GuessPass<true> << <dim3(qw / p0, qh / p0, 1), dim3(p0, p0, 1) >> >(qw, qh, a, b);
	else k_GuessPass<false> << <dim3(qw / p0, qh / p0, 1), dim3(p0, p0, 1) >> >(qw, qh, a, b);
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
	res = traceRay(r);
}

TraceResult TracerBase::TraceSingleRay(Ray r, DynamicScene* s)
{
	UpdateKernel(s);
	return traceRay(r);
}

CUDA_DEVICE unsigned int g_ShotRays;
CUDA_DEVICE unsigned int g_SuccRays;

__global__ void estimateLightVisibility(int w, int h, float scx, float scy, int recursion_depth)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	auto rng = g_SamplerData(y * w + x);
	if (x < w && y < h)
	{
		NormalizedT<Ray> r = g_SceneData.GenerateSensorRay(float(x * scx), float(y * scy));
		TraceResult r2;
		r2.Init();
		int d = -1;
		unsigned int N = 0, S = 0;
		while (traceRay(r.dir(), r.ori(), &r2) && ++d < recursion_depth)
		{
			DifferentialGeometry dg;
			BSDFSamplingRecord bRec(dg);
			r2.getBsdfSample(r, bRec, ETransportMode::ERadiance);

			for (int i = 0; i < g_SceneData.m_numLights; i++, N++)
			{
				PositionSamplingRecord pRec;
				g_SceneData.getLight(i)->samplePosition(pRec, rng.randomFloat2());
				S += V(pRec.p, dg.P);
			}

			r2.getMat().bsdf.sample(bRec, rng.randomFloat2());
			r = NormalizedT<Ray>(dg.P, bRec.getOutgoing());
		}
		atomicAdd(&g_ShotRays, N);
		atomicAdd(&g_SuccRays, S);
	}
}

float TracerBase::GetLightVisibility(DynamicScene* s, int recursion_depth)
{
	unsigned int zero = 0;
	ThrowCudaErrors(cudaMemcpyToSymbol(g_ShotRays, &zero, sizeof(unsigned int)));
	ThrowCudaErrors(cudaMemcpyToSymbol(g_SuccRays, &zero, sizeof(unsigned int)));
	UpdateKernel(s);

	int qw = 128, qh = 128, p0 = 16;
	float a = (float)s->getCamera()->As()->m_resolution.x / qw, b = (float)s->getCamera()->As()->m_resolution.y / qh;
	estimateLightVisibility << <dim3(qw / p0, qh / p0, 1), dim3(p0, p0, 1) >> >(qw, qh, a, b, recursion_depth);
	ThrowCudaErrors(cudaDeviceSynchronize());
	unsigned int N, S;
	ThrowCudaErrors(cudaMemcpyFromSymbol(&N, g_ShotRays, sizeof(unsigned int)));
	ThrowCudaErrors(cudaMemcpyFromSymbol(&S, g_SuccRays, sizeof(unsigned int)));
	return float(S) / float(N);
}

__global__ void depthKernel(DeviceDepthImage dImg)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x < dImg.w && y < dImg.h)
	{
		Ray r = g_SceneData.GenerateSensorRay(x, y);
		TraceResult r2 = traceRay(r);
		dImg.Store(x, y, r2.m_fDist);
	}
}

void TracerBase::RenderDepth(DeviceDepthImage dImg, DynamicScene* s)
{
	UpdateKernel(s);
	depthKernel << <dim3(dImg.w / 16 + 1, dImg.h / 16 + 1), dim3(16, 16) >> >(dImg);
	ThrowCudaErrors(cudaDeviceSynchronize());
}

}
