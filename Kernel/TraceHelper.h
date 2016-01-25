#pragma once
#include "TraceResult.h"
#include <Base/CudaRandom.h>
#include <Engine/KernelDynamicScene.h>

namespace CudaTracerLib {

extern CUDA_ALIGN(16) CUDA_CONST KernelDynamicScene g_SceneDataDevice;
extern CUDA_ALIGN(16) CUDA_DEVICE unsigned int g_RayTracedCounterDevice;
extern CUDA_ALIGN(16) CUDA_CONST CudaRNGBuffer g_RNGDataDevice;

extern CUDA_ALIGN(16) KernelDynamicScene g_SceneDataHost;
extern CUDA_ALIGN(16) unsigned int g_RayTracedCounterHost;
extern CUDA_ALIGN(16) CudaRNGBuffer g_RNGDataHost;

#ifdef ISCUDA
#define g_SceneData g_SceneDataDevice
#define g_RayTracedCounter g_RayTracedCounterDevice
#define g_RNGData g_RNGDataDevice
#else
#define g_SceneData g_SceneDataHost
#define g_RayTracedCounter g_RayTracedCounterHost
#define g_RNGData g_RNGDataHost
#endif

__device__ __host__ bool traceRay(const Vec3f& dir, const Vec3f& ori, TraceResult* a_Result);

CUDA_FUNC_IN TraceResult traceRay(const Ray& r)
{
	TraceResult r2;
	r2.Init();
	traceRay(r.dir(), r.ori(), &r2);
	return r2;
}

CUDA_DEVICE CUDA_HOST void fillDG(const Vec2f& bary, unsigned int triIdx, unsigned int nodeIdx, DifferentialGeometry& dg);

void k_INITIALIZE(DynamicScene* a_Scene, const CudaRNGBuffer& a_RngBuf);

unsigned int k_getNumRaysTraced();
void k_setNumRaysTraced(unsigned int i);

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

void __internal__IntersectBuffers(int N, traversalRay* a_RayBuffer, traversalResult* a_ResBuffer, bool SKIP_OUTER, bool ANY_HIT);

}