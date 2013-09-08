#pragma once

//__forceinline__
#define CUDA_INLINE inline

#ifdef __CUDACC__
#define CUDA_FUNC inline __host__ __device__
#define CUDA_FUNC_IN CUDA_INLINE __host__ __device__
#define CUDA_ONLY_FUNC __device__ CUDA_INLINE
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#define CUDA_CONST __constant__
#define CUDA_SHARED __shared__
#define CUDA_GLOBAL __global__
#define CUDA_LOCAL	__local__
#define CUDA_VIRTUAL __device__ virtual
#else
#define CUDA_FUNC inline
#define CUDA_FUNC_IN inline
#define CUDA_ONLY_FUNC inline
#define CUDA_HOST
#define CUDA_DEVICE
#define CUDA_CONST
#define CUDA_SHARED
#define CUDA_GLOBAL
#define CUDA_LOCAL
#define CUDA_VIRTUAL virtual
#endif

#ifdef __CUDA_ARCH__  
#define ISCUDA
#endif

#if defined(__CUDACC__) // NVCC
   #define CUDA_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
  #define CUDA_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
  #define CUDA_ALIGN(n) __declspec(align(n))
#else
  #error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

#define TYPE_FUNC(name) \
	static CUDA_FUNC_IN unsigned int TYPE() \
	{ \
		return name##_TYPE; \
	}

#ifdef __CUDACC__
 CUDA_FUNC_IN int getGlobalIdx_2D_2D()
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x; 
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}

CUDA_FUNC_IN int getGlobalIdx_3D_3D()
{
	int blockId = blockIdx.x 
			 + blockIdx.y * gridDim.x 
			 + gridDim.x * gridDim.y * blockIdx.z; 
	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
			  + (threadIdx.z * (blockDim.x * blockDim.y))
			  + (threadIdx.y * blockDim.x)
			  + threadIdx.x;
	return threadId;
}

#define threadId getGlobalIdx_2D_2D()
#define threadId_Unsafe threadId
#else
#define threadId 0
#include <Windows.h>
#define threadId_Unsafe GetCurrentThreadId()
#endif

#pragma warning(disable: 4482)
#pragma warning(disable: 4244)
#pragma warning(disable: 4800)
#pragma warning(disable: 4996)
#pragma warning(disable: 4305)