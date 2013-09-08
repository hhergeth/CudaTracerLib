#pragma once
#include "..\Engine\e_Camera.h"
#include "..\Engine\e_DynamicScene.h"

#ifdef __CUDACC__ 
extern texture<float4, 1> t_nodesA;
extern texture<float4, 1> t_tris;
extern texture<int,  1>   t_triIndices;
extern texture<float4, 1> t_SceneNodes;
extern texture<float4, 1> t_NodeTransforms;
extern texture<float4, 1> t_NodeInvTransforms;
#endif

template<typename T> CUDA_ONLY_FUNC void swapDevice(T& a, T& b)
{
	T q = a;
	a = b;
	b = q;
}

enum
{
    MaxBlockHeight      = 6,            // Upper bound for blockDim.y.
    EntrypointSentinel  = 0x76543210,   // Bottom-most stack entry, indicating the end of traversal.
};

extern CUDA_ALIGN(16) CUDA_CONST e_KernelDynamicScene g_SceneDataDevice;
extern CUDA_ALIGN(16) CUDA_DEVICE unsigned int g_RayTracedCounterDevice;
extern CUDA_ALIGN(16) CUDA_CONST e_CameraData g_CameraDataDevice;
extern CUDA_ALIGN(16) CUDA_CONST k_TracerRNGBuffer g_RNGDataDevice;

extern CUDA_ALIGN(16) e_KernelDynamicScene g_SceneDataHost;
extern CUDA_ALIGN(16) volatile LONG g_RayTracedCounterHost;
extern CUDA_ALIGN(16) e_CameraData g_CameraDataHost;
extern CUDA_ALIGN(16) k_TracerRNGBuffer g_RNGDataHost;

#ifdef ISCUDA
#define g_SceneData g_SceneDataDevice
#define g_RayTracedCounter g_RayTracedCounterDevice
#define g_CameraData g_CameraDataDevice
#define g_RNGData g_RNGDataDevice
#else
#define g_SceneData g_SceneDataHost
#define g_RayTracedCounter g_RayTracedCounterHost
#define g_CameraData g_CameraDataHost
#define g_RNGData g_RNGDataHost
#endif

#ifdef __CUDACC__
#define k_TracerBase_update_TracedRays { cudaMemcpyFromSymbol(&m_uNumRaysTraced, g_RayTracedCounterDevice, sizeof(unsigned int)); }
#else
#define k_TracerBase_update_TracedRays { m_uNumRaysTraced = g_RayTracedCounterHost; }
#endif
	
#ifdef ISCUDA
	__device__
#else
	__host__
#endif
bool k_TraceRayNode(const float3& dir, const float3& ori, TraceResult* a_Result, const e_Node* N, int ln);


#ifdef ISCUDA
	__device__
#else
	__host__
#endif
bool k_TraceRay(const float3& dir, const float3& ori, TraceResult* a_Result);

CUDA_FUNC_IN TraceResult k_TraceRay(const Ray& r)
{
	TraceResult r2;
	r2.Init();
	k_TraceRay(r.direction, r.origin, &r2);
	return r2;
}

void k_INITIALIZE(const e_KernelDynamicScene& a_Data);
void k_STARTPASS(e_DynamicScene* a_Scene, e_Camera* a_Camera, const k_TracerRNGBuffer& a_RngBuf);