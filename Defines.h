#pragma once

#include <malloc.h>
#include <stdlib.h>

#define EXT_TRI
#define NUM_UV_SETS 1

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

 CUDA_FUNC_IN int getGlobalIdx_2D_2D()
{
#ifdef ISCUDA
	int blockId = blockIdx.x + blockIdx.y * gridDim.x; 
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
#else
	//return GetCurrentThreadId();
	return 0;
#endif
}

CUDA_FUNC_IN int getGlobalIdx_3D_3D()
{
#ifdef ISCUDA
	int blockId = blockIdx.x 
			 + blockIdx.y * gridDim.x 
			 + gridDim.x * gridDim.y * blockIdx.z; 
	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
			  + (threadIdx.z * (blockDim.x * blockDim.y))
			  + (threadIdx.y * blockDim.x)
			  + threadIdx.x;
	return threadId;
#else
	//return GetCurrentThreadId();
	return 0;
#endif
}

#define threadId getGlobalIdx_2D_2D()

#define DMAX2(A, B) ((A) > (B) ? (A) : (B))
#define DMAX3(A, B, C) DMAX2(DMAX2(A, B), C)
#define DMAX4(A, B, C, D) DMAX2(DMAX3(A, B, C), D)
#define DMAX5(A, B, C, D, E) DMAX2(DMAX4(A, B, C, D), E)
#define DMAX6(A, B, C, D, E, F) DMAX2(DMAX5(A, B, C, D, E), F)
#define DMAX7(A, B, C, D, E, F, G) DMAX2(DMAX6(A, B, C, D, E, F), G)
#define DMAX8(A, B, C, D, E, F, G, H) DMAX2(DMAX7(A, B, C, D, E, F, G), H)
#define DMAX9(A, B, C, D, E, F, G, H, I) DMAX2(DMAX8(A, B, C, D, E, F, G, H), I)

#define DMIN2(A, B) ((A) < (B) ? (A) : (B))
#define DMIN3(A, B, C) DMIN2(DMIN2(A, B), C)
#define DMIN4(A, B, C, D) DMIN2(DMIN3(A, B, C), D)
#define DMIN5(A, B, C, D, E) DMIN2(DMIN4(A, B, C, D), E)
#define DMIN6(A, B, C, D, E, F) DMIN2(DMIN5(A, B, C, D, E), F)
#define DMIN7(A, B, C, D, E, F, G) DMIN2(DMIN6(A, B, C, D, E, F), G)
#define DMIN8(A, B, C, D, E, F, G, H) DMIN2(DMIN7(A, B, C, D, E, F, G), H)
#define DMIN9(A, B, C, D, E, F, G, H, I) DMIN2(DMIN8(A, B, C, D, E, F, G, H), I)

#define RND_UP(VAL, MOD) (VAL + (((VAL) % (MOD)) != 0 ? ((MOD) - ((VAL) % (MOD))) : (0)))
#define RND_16(VAL) RND_UP(VAL, 16)

inline void CudaSetToZero(void* dest, size_t length)
{
	static void* zeroBuf = 0;
	static size_t zeroBufLength = 0;
	if(!zeroBuf || zeroBufLength < length)
	{
		if(zeroBuf)
			free(zeroBuf);
		zeroBufLength = RND_16(DMAX2(length, zeroBufLength));
		zeroBuf = malloc(zeroBufLength);
		for(int i = 0; i < zeroBufLength / 8; i++)
			*((unsigned long long*)zeroBuf + i) = 0;
	}
	cudaMemcpy(dest, zeroBuf, length, cudaMemcpyHostToDevice);
}
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

#define CALL_TYPE(t,func) \
	case t##_TYPE : \
		return ((t*)Data)->func;
#define CALL_FUNC1(_TYPE0_, func) \
	switch (type) \
	{ \
		CALL_TYPE(_TYPE0_, func) \
	}
#define CALL_FUNC2(_TYPE0_,_TYPE1_, func) \
	switch (type) \
	{ \
		CALL_TYPE(_TYPE0_, func) \
		CALL_TYPE(_TYPE1_, func) \
	}
#define CALL_FUNC3(_TYPE0_,_TYPE1_,_TYPE2_, func) \
	switch (type) \
	{ \
		CALL_TYPE(_TYPE0_, func) \
		CALL_TYPE(_TYPE1_, func) \
		CALL_TYPE(_TYPE2_, func) \
	}
#define CALL_FUNC4(_TYPE0_,_TYPE1_,_TYPE2_,_TYPE3_, func) \
	switch (type) \
	{ \
		CALL_TYPE(_TYPE0_, func) \
		CALL_TYPE(_TYPE1_, func) \
		CALL_TYPE(_TYPE2_, func) \
		CALL_TYPE(_TYPE3_, func) \
	}
#define CALL_FUNC5(_TYPE0_,_TYPE1_,_TYPE2_,_TYPE3_,_TYPE4_, func) \
	switch (type) \
	{ \
		CALL_TYPE(_TYPE0_, func) \
		CALL_TYPE(_TYPE1_, func) \
		CALL_TYPE(_TYPE2_, func) \
		CALL_TYPE(_TYPE3_, func) \
		CALL_TYPE(_TYPE4_, func) \
	}

#define CALL_FUNC6(_TYPE0_,_TYPE1_,_TYPE2_,_TYPE3_,_TYPE4_,_TYPE5_, func) \
	switch (type) \
	{ \
		CALL_TYPE(_TYPE0_, func) \
		CALL_TYPE(_TYPE1_, func) \
		CALL_TYPE(_TYPE2_, func) \
		CALL_TYPE(_TYPE3_, func) \
		CALL_TYPE(_TYPE4_, func) \
		CALL_TYPE(_TYPE5_, func) \
	}
#define CALL_FUNC7(_TYPE0_,_TYPE1_,_TYPE2_,_TYPE3_,_TYPE4_,_TYPE5_,_TYPE6_, func) \
	switch (type) \
	{ \
		CALL_TYPE(_TYPE0_, func) \
		CALL_TYPE(_TYPE1_, func) \
		CALL_TYPE(_TYPE2_, func) \
		CALL_TYPE(_TYPE3_, func) \
		CALL_TYPE(_TYPE4_, func) \
		CALL_TYPE(_TYPE5_, func) \
		CALL_TYPE(_TYPE6_, func) \
	}
#define CALL_FUNC8(_TYPE0_,_TYPE1_,_TYPE2_,_TYPE3_,_TYPE4_,_TYPE5_,_TYPE6_,_TYPE7_, func) \
	switch (type) \
	{ \
		CALL_TYPE(_TYPE0_, func) \
		CALL_TYPE(_TYPE1_, func) \
		CALL_TYPE(_TYPE2_, func) \
		CALL_TYPE(_TYPE3_, func) \
		CALL_TYPE(_TYPE4_, func) \
		CALL_TYPE(_TYPE5_, func) \
		CALL_TYPE(_TYPE6_, func) \
		CALL_TYPE(_TYPE7_, func) \
	}
#define CALL_FUNC9(_TYPE0_,_TYPE1_,_TYPE2_,_TYPE3_,_TYPE4_,_TYPE5_,_TYPE6_,_TYPE7_,_TYPE8_, func) \
	switch (type) \
	{ \
		CALL_TYPE(_TYPE0_, func) \
		CALL_TYPE(_TYPE1_, func) \
		CALL_TYPE(_TYPE2_, func) \
		CALL_TYPE(_TYPE3_, func) \
		CALL_TYPE(_TYPE4_, func) \
		CALL_TYPE(_TYPE5_, func) \
		CALL_TYPE(_TYPE6_, func) \
		CALL_TYPE(_TYPE7_, func) \
		CALL_TYPE(_TYPE8_, func) \
	}
#define CALL_FUNC10(_TYPE0_,_TYPE1_,_TYPE2_,_TYPE3_,_TYPE4_,_TYPE5_,_TYPE6_,_TYPE7_,_TYPE8_,_TYPE9_, func) \
	switch (type) \
	{ \
		CALL_TYPE(_TYPE0_, func) \
		CALL_TYPE(_TYPE1_, func) \
		CALL_TYPE(_TYPE2_, func) \
		CALL_TYPE(_TYPE3_, func) \
		CALL_TYPE(_TYPE4_, func) \
		CALL_TYPE(_TYPE5_, func) \
		CALL_TYPE(_TYPE6_, func) \
		CALL_TYPE(_TYPE7_, func) \
		CALL_TYPE(_TYPE8_, func) \
		CALL_TYPE(_TYPE9_, func) \
	}
#define CALL_FUNC11(_TYPE0_,_TYPE1_,_TYPE2_,_TYPE3_,_TYPE4_,_TYPE5_,_TYPE6_,_TYPE7_,_TYPE8_,_TYPE9_,_TYPE10_, func) \
	switch (type) \
	{ \
		CALL_TYPE(_TYPE0_, func) \
		CALL_TYPE(_TYPE1_, func) \
		CALL_TYPE(_TYPE2_, func) \
		CALL_TYPE(_TYPE3_, func) \
		CALL_TYPE(_TYPE4_, func) \
		CALL_TYPE(_TYPE5_, func) \
		CALL_TYPE(_TYPE6_, func) \
		CALL_TYPE(_TYPE7_, func) \
		CALL_TYPE(_TYPE8_, func) \
		CALL_TYPE(_TYPE9_, func) \
		CALL_TYPE(_TYPE10_, func) \
	}
#define CALL_FUNC12(_TYPE0_,_TYPE1_,_TYPE2_,_TYPE3_,_TYPE4_,_TYPE5_,_TYPE6_,_TYPE7_,_TYPE8_,_TYPE9_,_TYPE10_,_TYPE11_, func) \
	switch (type) \
	{ \
		CALL_TYPE(_TYPE0_, func) \
		CALL_TYPE(_TYPE1_, func) \
		CALL_TYPE(_TYPE2_, func) \
		CALL_TYPE(_TYPE3_, func) \
		CALL_TYPE(_TYPE4_, func) \
		CALL_TYPE(_TYPE5_, func) \
		CALL_TYPE(_TYPE6_, func) \
		CALL_TYPE(_TYPE7_, func) \
		CALL_TYPE(_TYPE8_, func) \
		CALL_TYPE(_TYPE9_, func) \
		CALL_TYPE(_TYPE10_, func) \
		CALL_TYPE(_TYPE11_, func) \
	}
#define CALL_FUNC13(_TYPE0_,_TYPE1_,_TYPE2_,_TYPE3_,_TYPE4_,_TYPE5_,_TYPE6_,_TYPE7_,_TYPE8_,_TYPE9_,_TYPE10_,_TYPE11_,_TYPE12_, func) \
	switch (type) \
	{ \
		CALL_TYPE(_TYPE0_, func) \
		CALL_TYPE(_TYPE1_, func) \
		CALL_TYPE(_TYPE2_, func) \
		CALL_TYPE(_TYPE3_, func) \
		CALL_TYPE(_TYPE4_, func) \
		CALL_TYPE(_TYPE5_, func) \
		CALL_TYPE(_TYPE6_, func) \
		CALL_TYPE(_TYPE7_, func) \
		CALL_TYPE(_TYPE8_, func) \
		CALL_TYPE(_TYPE9_, func) \
		CALL_TYPE(_TYPE10_, func) \
		CALL_TYPE(_TYPE11_, func) \
		CALL_TYPE(_TYPE12_, func) \
	}
#define CALL_FUNC14(_TYPE0_,_TYPE1_,_TYPE2_,_TYPE3_,_TYPE4_,_TYPE5_,_TYPE6_,_TYPE7_,_TYPE8_,_TYPE9_,_TYPE10_,_TYPE11_,_TYPE12_,_TYPE13_, func) \
	switch (type) \
	{ \
		CALL_TYPE(_TYPE0_, func) \
		CALL_TYPE(_TYPE1_, func) \
		CALL_TYPE(_TYPE2_, func) \
		CALL_TYPE(_TYPE3_, func) \
		CALL_TYPE(_TYPE4_, func) \
		CALL_TYPE(_TYPE5_, func) \
		CALL_TYPE(_TYPE6_, func) \
		CALL_TYPE(_TYPE7_, func) \
		CALL_TYPE(_TYPE8_, func) \
		CALL_TYPE(_TYPE9_, func) \
		CALL_TYPE(_TYPE10_, func) \
		CALL_TYPE(_TYPE11_, func) \
		CALL_TYPE(_TYPE12_, func) \
		CALL_TYPE(_TYPE13_, func) \
	}

#define CALL_FUNC15(_TYPE0_,_TYPE1_,_TYPE2_,_TYPE3_,_TYPE4_,_TYPE5_,_TYPE6_,_TYPE7_,_TYPE8_,_TYPE9_,_TYPE10_,_TYPE11_,_TYPE12_,_TYPE13_,_TYPE14_, func) \
	switch (type) \
	{ \
		CALL_TYPE(_TYPE0_, func) \
		CALL_TYPE(_TYPE1_, func) \
		CALL_TYPE(_TYPE2_, func) \
		CALL_TYPE(_TYPE3_, func) \
		CALL_TYPE(_TYPE4_, func) \
		CALL_TYPE(_TYPE5_, func) \
		CALL_TYPE(_TYPE6_, func) \
		CALL_TYPE(_TYPE7_, func) \
		CALL_TYPE(_TYPE8_, func) \
		CALL_TYPE(_TYPE9_, func) \
		CALL_TYPE(_TYPE10_, func) \
		CALL_TYPE(_TYPE11_, func) \
		CALL_TYPE(_TYPE12_, func) \
		CALL_TYPE(_TYPE13_, func) \
		CALL_TYPE(_TYPE14_, func) \
	}

#define CALL_FUNC16(_TYPE0_,_TYPE1_,_TYPE2_,_TYPE3_,_TYPE4_,_TYPE5_,_TYPE6_,_TYPE7_,_TYPE8_,_TYPE9_,_TYPE10_,_TYPE11_,_TYPE12_,_TYPE13_,_TYPE14_,_TYPE15_, func) \
	switch (type) \
	{ \
		CALL_TYPE(_TYPE0_, func) \
		CALL_TYPE(_TYPE1_, func) \
		CALL_TYPE(_TYPE2_, func) \
		CALL_TYPE(_TYPE3_, func) \
		CALL_TYPE(_TYPE4_, func) \
		CALL_TYPE(_TYPE5_, func) \
		CALL_TYPE(_TYPE6_, func) \
		CALL_TYPE(_TYPE7_, func) \
		CALL_TYPE(_TYPE8_, func) \
		CALL_TYPE(_TYPE9_, func) \
		CALL_TYPE(_TYPE10_, func) \
		CALL_TYPE(_TYPE11_, func) \
		CALL_TYPE(_TYPE12_, func) \
		CALL_TYPE(_TYPE13_, func) \
		CALL_TYPE(_TYPE15_, func) \
		CALL_TYPE(_TYPE16_, func) \
	}

typedef char e_String[256];

//thats not const correct

struct e_BaseType
{
	virtual void Update()
	{
	}
};

template<typename BaseType, int Size> struct e_AggregateBaseType
{
	unsigned int type;
	CUDA_ALIGN(16) unsigned char Data[Size];

	template<typename SpecializedType> CUDA_FUNC_IN SpecializedType* As() const
	{
		return (SpecializedType*)Data;
	}

	template<typename SpecializedType> CUDA_FUNC_IN void SetData(const SpecializedType& val)
	{
		memcpy(Data, &val, sizeof(SpecializedType));
		type = SpecializedType::TYPE();
	}

	template<typename SpecializedType> CUDA_FUNC_IN bool Is() const
	{
		return type == SpecializedType::TYPE();
	}

	CUDA_FUNC_IN BaseType* As() const
	{
		return As<BaseType>();
	}
};

#pragma warning(disable: 4482)
#pragma warning(disable: 4244)
#pragma warning(disable: 4800)
#pragma warning(disable: 4996)
#pragma warning(disable: 4305)
#pragma warning(disable: 4204)

template<typename T> class e_Variable
{
public:
	T* host, *device;
#ifdef __CUDACC__
	e_Variable()
	{
	}
#else
	e_Variable()
		: host(0), device(0)
	{
	}
#endif
	/*
	template<typename U, typename V> CUDA_HOST e_Variable(e_BufferReference<U, V> r)
	{
		host = (T*)r.operator->();
		device = (T*)r.getDevice();
	}*/
	e_Variable(T* h, T* d)
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
};