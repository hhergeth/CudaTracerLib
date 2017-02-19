#pragma once

#include <malloc.h>
#include <stdlib.h>
#include <string>
#include <memory>

namespace CudaTracerLib {

#define EXT_TRI
#define NUM_UV_SETS 1
#define MAX_AREALIGHT_NUM 2

#ifdef _MSC_VER
#define ISWINDOWS
#endif

#ifdef CTL_EXPORT_SYMBOLS
#define CTL_EXPORT __declspec(dllexport)
#endif

#ifdef CTL_IMPORT_SYMBOLS
#define CTL_EXPORT __declspec(dllimport)
#endif

#ifndef CTL_EXPORT
#define CTL_EXPORT
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

#ifndef __func__
#define __func__ __FUNCTION__
#endif

#if !defined(CUDA_RELEASE_BUILD)
#if __CUDACC__
#define CTL_ASSERT(X) ((X) ? ((void)0) : (void)printf("Assertion failed!\n%s:%d\n%s", __FILE__, __LINE__, #X))
#else
#define CTL_ASSERT(X) ((X) ? ((void)0) : throw std::runtime_error(format("Assertion failed!\n%s:%d\n%s", __FILE__, __LINE__, #X)))
#endif
#else
//evaluate the expression either way, it possibly has side effects
	CUDA_FUNC_IN void noop() {}
#   define CTL_ASSERT(X) ((X) ? noop() : noop())
#endif

//code is from this great answer : http://stackoverflow.com/a/26221725/1715849
template<typename ... Args> std::string format(const std::string& format, Args ... args)
{
	size_t size = snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
	std::unique_ptr<char[]> buf(new char[size]);
	snprintf(buf.get(), size, format.c_str(), args ...);
	return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

//http://stackoverflow.com/questions/12778949/cuda-memory-alignment
//credit to harrsim!
#if defined(__CUDACC__) // NVCC
#define CUDA_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define CUDA_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define CUDA_ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

CTL_EXPORT void __ThrowCudaErrors__(const char* file, int line, ...);
#define ThrowCudaErrors(...) __ThrowCudaErrors__(__FILE__, __LINE__, ##__VA_ARGS__, -1)

template<typename T> CUDA_FUNC_IN void swapk(T& a, T& b)
{
	T q = a;
	a = b;
	b = q;
}

CUDA_FUNC_IN unsigned int getGlobalIdx_2D_2D()
{
#ifdef ISCUDA
	unsigned int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
#else
	return 0u;
#endif
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

CTL_EXPORT void CudaSetToZero(void* dest, size_t length);
CTL_EXPORT void CudaSetToZero_FreeBuffer();
template<typename T> inline void ZeroMemoryCuda(T* cudaVar)
{
	CudaSetToZero(cudaVar, sizeof(T));
}
#define ZeroSymbol(SYMBOL) \
	{ \
		void* tar = 0; \
		ThrowCudaErrors(cudaGetSymbolAddress(&tar, SYMBOL)); \
		CudaSetToZero(tar, sizeof(SYMBOL)); \
	}

#define CopyToSymbol(SYMBOL, value) \
	{ \
		void* tar = 0; \
		ThrowCudaErrors(cudaGetSymbolAddress(&tar, SYMBOL)); \
		ThrowCudaErrors(cudaMemcpy(tar, &value, sizeof(value), cudaMemcpyHostToDevice)); \
	}

#define CopyFromSymbol(value, SYMBOL) \
	{ \
		void* tar = 0; \
		ThrowCudaErrors(cudaGetSymbolAddress(&tar, SYMBOL)); \
		cudaMemcpy(&value, tar, sizeof(value), cudaMemcpyDeviceToHost); \
	}

template<typename T> struct CudaStaticWrapper
{
protected:
	CUDA_ALIGN(256) unsigned char m_data[sizeof(T)];
public:
	CUDA_FUNC_IN CudaStaticWrapper()
	{

	}
	CUDA_FUNC_IN operator const T& () const
	{
		return As();
	}
	CUDA_FUNC_IN operator T& ()
	{
		return As();
	}
	CUDA_FUNC_IN T* operator->()
	{
		return &As();
	}
	CUDA_FUNC_IN const T* operator->() const
	{
		return &As();
	}
	CUDA_FUNC_IN T& operator*()
	{
		return As();
	}
	CUDA_FUNC_IN const T& operator*() const
	{
		return As();
	}
	CUDA_FUNC_IN const T& As() const
	{
		return *(T*)m_data;
	}
	CUDA_FUNC_IN T& As()
	{
		return *(T*)m_data;
	}
};

template<typename T> class e_Variable
{
public:
	CUDA_ALIGN(16) T* host;
	CUDA_ALIGN(16) T* device;
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
	template<typename U> CUDA_FUNC_IN e_Variable<U> As() const
	{
		return e_Variable<U>((U*)host, (U*)device);
	}
};

}
