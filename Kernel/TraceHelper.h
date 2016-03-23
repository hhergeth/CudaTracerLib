#pragma once
#include "TraceResult.h"
#include <Base/CudaRandom.h>
#include <Engine/KernelDynamicScene.h>
#include "Sampler_device.h"

namespace CudaTracerLib {

extern CUDA_ALIGN(16) CUDA_CONST KernelDynamicScene g_SceneDataDevice;
extern CUDA_ALIGN(16) CUDA_DEVICE unsigned int g_RayTracedCounterDevice;
extern CUDA_ALIGN(16) CUDA_CONST SamplerData g_SamplerDataDevice;
extern CUDA_CONST unsigned int g_PassIndexDevice;

CTL_EXPORT extern CUDA_ALIGN(16) KernelDynamicScene g_SceneDataHost;
CTL_EXPORT extern CUDA_ALIGN(16) unsigned int g_RayTracedCounterHost;
CTL_EXPORT extern CUDA_ALIGN(16) SamplerData g_SamplerDataHost;
CTL_EXPORT extern unsigned int g_PassIndexHost;

#ifdef ISCUDA
#define g_SceneData g_SceneDataDevice
#define g_RayTracedCounter g_RayTracedCounterDevice
#define g_SamplerData g_SamplerDataDevice
#define g_PassIndex g_PassIndexDevice
#else
#define g_SceneData g_SceneDataHost
#define g_RayTracedCounter g_RayTracedCounterHost
#define g_SamplerData g_SamplerDataHost
#define g_PassIndex g_PassIndexHost
#endif

CTL_EXPORT CUDA_DEVICE CUDA_HOST bool traceRay(const Vec3f& dir, const Vec3f& ori, TraceResult* a_Result);

CUDA_FUNC_IN TraceResult traceRay(const Ray& r)
{
	TraceResult r2;
	r2.Init();
	traceRay(r.dir(), r.ori(), &r2);
	return r2;
}

CTL_EXPORT CUDA_DEVICE CUDA_HOST void fillDG(const Vec2f& bary, unsigned int triIdx, unsigned int nodeIdx, DifferentialGeometry& dg);

CTL_EXPORT void InitializeKernel();
CTL_EXPORT void DeinitializeKernel();

CTL_EXPORT void UpdateKernel(DynamicScene* a_Scene, ISamplingSequenceGenerator* sampler = 0, const unsigned int* passIdx = 0);
CTL_EXPORT void UpdateSamplerData(unsigned int numSequences);

CTL_EXPORT unsigned int k_getNumRaysTraced();
CTL_EXPORT void k_setNumRaysTraced(unsigned int i);

struct traversalRay
{
	Vec4f a;
	Vec4f b;
};

struct CUDA_ALIGN(16) traversalResult
{
	float dist;
	int nodeIdx;
	int triIdx;
	int bCoords;//half2
	CUDA_DEVICE CUDA_HOST void toResult(TraceResult* tR, KernelDynamicScene& g_SceneData);
};

CTL_EXPORT void __internal__IntersectBuffers(int N, traversalRay* a_RayBuffer, traversalResult* a_ResBuffer, bool SKIP_OUTER, bool ANY_HIT);

}