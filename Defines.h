#pragma once

#include <malloc.h>
#include <stdlib.h>

namespace CudaTracerLib {

#define EXT_TRI
#define NUM_UV_SETS 1
#define MAX_AREALIGHT_NUM 2

#ifdef _MSC_VER
#define ISWINDOWS
#else
#define ISUNIX
#endif

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

#if _DEBUG
#if __CUDACC__
#define CT_ASSERT(X) ((X) ? ((void)0) : printf("Assertion failed!\n%s:%d\n%s", __FILE__, __LINE__, #X))
#else
#define CT_ASSERT(X) ((X) ? ((void)0) : fail("Assertion failed!\n%s:%d\n%s", __FILE__, __LINE__, #X))
#endif
#else
#   define CT_ASSERT(X) ((void)0)
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

void fail(const char* format, ...);

void __ThrowCudaErrors__(const char* file, int line, ...);
#define ThrowCudaErrors(...) __ThrowCudaErrors__(__FILE__, __LINE__, __VA_ARGS__, -1)

template<typename T> CUDA_FUNC_IN void swapk(T* a, T* b)
{
	T q = *a;
	*a = *b;
	*b = q;
}

template<typename T> CUDA_FUNC_IN void swapk(T& a, T& b)
{
	T q = a;
	a = b;
	b = q;
}

#define Dmax2(A, B) ((A) > (B) ? (A) : (B))
#define Dmax3(A, B, C) Dmax2(Dmax2(A, B), C)
#define Dmax4(A, B, C, D) Dmax2(Dmax3(A, B, C), D)
#define Dmax5(A, B, C, D, E) Dmax2(Dmax4(A, B, C, D), E)
#define Dmax6(A, B, C, D, E, F) Dmax2(Dmax5(A, B, C, D, E), F)
#define Dmax7(A, B, C, D, E, F, G) Dmax2(Dmax6(A, B, C, D, E, F), G)
#define Dmax8(A, B, C, D, E, F, G, H) Dmax2(Dmax7(A, B, C, D, E, F, G), H)
#define Dmax9(A, B, C, D, E, F, G, H, I) Dmax2(Dmax8(A, B, C, D, E, F, G, H), I)

#define Dmin2(A, B) ((A) < (B) ? (A) : (B))
#define Dmin3(A, B, C) Dmin2(Dmin2(A, B), C)
#define Dmin4(A, B, C, D) Dmin2(Dmin3(A, B, C), D)
#define Dmin5(A, B, C, D, E) Dmin2(Dmin4(A, B, C, D), E)
#define Dmin6(A, B, C, D, E, F) Dmin2(Dmin5(A, B, C, D, E), F)
#define Dmin7(A, B, C, D, E, F, G) Dmin2(Dmin6(A, B, C, D, E, F), G)
#define Dmin8(A, B, C, D, E, F, G, H) Dmin2(Dmin7(A, B, C, D, E, F, G), H)
#define Dmin9(A, B, C, D, E, F, G, H, I) Dmin2(Dmin8(A, B, C, D, E, F, G, H), I)

#define RND_UP(VAL, MOD) (VAL + (((VAL) % (MOD)) != 0 ? ((MOD) - ((VAL) % (MOD))) : (0)))
#define RND_16(VAL) RND_UP(VAL, 16)

void CudaSetToZero(void* dest, size_t length);
template<typename T> inline void ZeroMemoryCuda(T* cudaVar)
{
	CudaSetToZero(cudaVar, sizeof(T));
}
#define ZeroSymbol(SYMBOL) \
	{ \
	void* tar = 0; \
	cudaError_t r = cudaGetSymbolAddress(&tar, SYMBOL); \
	CudaSetToZero(tar, sizeof(SYMBOL)); \
	}

#pragma warning(disable: 4244)

template<typename T> class e_Variable
{
public:
	T* host, *device;
	CUDA_FUNC_IN e_Variable()
	{
	}
	/*
	template<typename U, typename V> CUDA_HOST e_Variable(BufferReference<U, V> r)
	{
	host = (T*)r.operator->();
	device = (T*)r.getDevice();
	}*/
	CUDA_FUNC_IN e_Variable(T* h, T* d)
		: host(h), device(d)
	{

	}
	CUDA_FUNC_IN T& operator[](unsigned int i) const
	{
#ifdef ISCUDA
		return device[i];
#else
		return host[i];
#endif
	}
	CUDA_FUNC_IN T* operator->() const
	{
#ifdef ISCUDA
		return device;
#else
		return host;
#endif
	}
	CUDA_FUNC_IN T* operator*() const
	{
#ifdef ISCUDA
		return device;
#else
		return host;
#endif
	}
	template<typename T> CUDA_FUNC_IN e_Variable<T> As() const
	{
		return e_Variable<T>((T*)host, (T*)device);
	}
};

}