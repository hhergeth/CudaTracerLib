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

#if _DEBUG && !__CUDACC__
#   define CT_ASSERT(X) ((X) ? ((void)0) : fail("Assertion failed!\n%s:%d\n%s", __FILE__, __LINE__, #X))
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

#define CALL_TYPE(t,func) \
	if(type == t::TYPE()) \
		return ((t*)Data)->func;
#define CALL_FUNC1(_TYPE0_, func) \
	CALL_TYPE(_TYPE0_, func)
#define CALL_FUNC2(_TYPE0_,_TYPE1_, func) \
	CALL_TYPE(_TYPE0_, func) \
	CALL_TYPE(_TYPE1_, func)
#define CALL_FUNC3(_TYPE0_,_TYPE1_,_TYPE2_, func) \
	CALL_TYPE(_TYPE0_, func) \
	CALL_TYPE(_TYPE1_, func) \
	CALL_TYPE(_TYPE2_, func)
#define CALL_FUNC4(_TYPE0_,_TYPE1_,_TYPE2_,_TYPE3_, func) \
	CALL_TYPE(_TYPE0_, func) \
	CALL_TYPE(_TYPE1_, func) \
	CALL_TYPE(_TYPE2_, func) \
	CALL_TYPE(_TYPE3_, func)
#define CALL_FUNC5(_TYPE0_,_TYPE1_,_TYPE2_,_TYPE3_,_TYPE4_, func) \
	CALL_TYPE(_TYPE0_, func) \
	CALL_TYPE(_TYPE1_, func) \
	CALL_TYPE(_TYPE2_, func) \
	CALL_TYPE(_TYPE3_, func) \
	CALL_TYPE(_TYPE4_, func)

#define CALL_FUNC6(_TYPE0_,_TYPE1_,_TYPE2_,_TYPE3_,_TYPE4_,_TYPE5_, func) \
	CALL_TYPE(_TYPE0_, func) \
	CALL_TYPE(_TYPE1_, func) \
	CALL_TYPE(_TYPE2_, func) \
	CALL_TYPE(_TYPE3_, func) \
	CALL_TYPE(_TYPE4_, func) \
	CALL_TYPE(_TYPE5_, func)
#define CALL_FUNC7(_TYPE0_,_TYPE1_,_TYPE2_,_TYPE3_,_TYPE4_,_TYPE5_,_TYPE6_, func) \
	CALL_TYPE(_TYPE0_, func) \
	CALL_TYPE(_TYPE1_, func) \
	CALL_TYPE(_TYPE2_, func) \
	CALL_TYPE(_TYPE3_, func) \
	CALL_TYPE(_TYPE4_, func) \
	CALL_TYPE(_TYPE5_, func) \
	CALL_TYPE(_TYPE6_, func)
#define CALL_FUNC8(_TYPE0_,_TYPE1_,_TYPE2_,_TYPE3_,_TYPE4_,_TYPE5_,_TYPE6_,_TYPE7_, func) \
	CALL_TYPE(_TYPE0_, func) \
	CALL_TYPE(_TYPE1_, func) \
	CALL_TYPE(_TYPE2_, func) \
	CALL_TYPE(_TYPE3_, func) \
	CALL_TYPE(_TYPE4_, func) \
	CALL_TYPE(_TYPE5_, func) \
	CALL_TYPE(_TYPE6_, func) \
	CALL_TYPE(_TYPE7_, func)
#define CALL_FUNC9(_TYPE0_,_TYPE1_,_TYPE2_,_TYPE3_,_TYPE4_,_TYPE5_,_TYPE6_,_TYPE7_,_TYPE8_, func) \
	CALL_TYPE(_TYPE0_, func) \
	CALL_TYPE(_TYPE1_, func) \
	CALL_TYPE(_TYPE2_, func) \
	CALL_TYPE(_TYPE3_, func) \
	CALL_TYPE(_TYPE4_, func) \
	CALL_TYPE(_TYPE5_, func) \
	CALL_TYPE(_TYPE6_, func) \
	CALL_TYPE(_TYPE7_, func) \
	CALL_TYPE(_TYPE8_, func)
#define CALL_FUNC10(_TYPE0_,_TYPE1_,_TYPE2_,_TYPE3_,_TYPE4_,_TYPE5_,_TYPE6_,_TYPE7_,_TYPE8_,_TYPE9_, func) \
	CALL_TYPE(_TYPE0_, func) \
	CALL_TYPE(_TYPE1_, func) \
	CALL_TYPE(_TYPE2_, func) \
	CALL_TYPE(_TYPE3_, func) \
	CALL_TYPE(_TYPE4_, func) \
	CALL_TYPE(_TYPE5_, func) \
	CALL_TYPE(_TYPE6_, func) \
	CALL_TYPE(_TYPE7_, func) \
	CALL_TYPE(_TYPE8_, func) \
	CALL_TYPE(_TYPE9_, func)
#define CALL_FUNC11(_TYPE0_,_TYPE1_,_TYPE2_,_TYPE3_,_TYPE4_,_TYPE5_,_TYPE6_,_TYPE7_,_TYPE8_,_TYPE9_,_TYPE10_, func) \
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
	CALL_TYPE(_TYPE10_, func)
#define CALL_FUNC12(_TYPE0_,_TYPE1_,_TYPE2_,_TYPE3_,_TYPE4_,_TYPE5_,_TYPE6_,_TYPE7_,_TYPE8_,_TYPE9_,_TYPE10_,_TYPE11_, func) \
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
	CALL_TYPE(_TYPE11_, func)
#define CALL_FUNC13(_TYPE0_,_TYPE1_,_TYPE2_,_TYPE3_,_TYPE4_,_TYPE5_,_TYPE6_,_TYPE7_,_TYPE8_,_TYPE9_,_TYPE10_,_TYPE11_,_TYPE12_, func) \
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
	CALL_TYPE(_TYPE12_, func)
#define CALL_FUNC14(_TYPE0_,_TYPE1_,_TYPE2_,_TYPE3_,_TYPE4_,_TYPE5_,_TYPE6_,_TYPE7_,_TYPE8_,_TYPE9_,_TYPE10_,_TYPE11_,_TYPE12_,_TYPE13_, func) \
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
	CALL_TYPE(_TYPE13_, func)

#define CALL_FUNC15(_TYPE0_,_TYPE1_,_TYPE2_,_TYPE3_,_TYPE4_,_TYPE5_,_TYPE6_,_TYPE7_,_TYPE8_,_TYPE9_,_TYPE10_,_TYPE11_,_TYPE12_,_TYPE13_,_TYPE14_, func) \
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
	CALL_TYPE(_TYPE14_, func)

#define CALL_FUNC16(_TYPE0_,_TYPE1_,_TYPE2_,_TYPE3_,_TYPE4_,_TYPE5_,_TYPE6_,_TYPE7_,_TYPE8_,_TYPE9_,_TYPE10_,_TYPE11_,_TYPE12_,_TYPE13_,_TYPE14_,_TYPE15_, func) \
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
	CALL_TYPE(_TYPE16_, func)

template<int ID> struct e_BaseTypeHelper
{
	static CUDA_FUNC_IN unsigned int BASE_TYPE()
	{
		return ID << 1;
	}
};

template<int ID> struct e_DerivedTypeHelper
{
	static CUDA_FUNC_IN unsigned int TYPE()
	{
		return ID;
	}
};

struct VC13_ClassLayouter
{
	virtual ~VC13_ClassLayouter() {}
};

struct e_BaseType : public VC13_ClassLayouter
{
	virtual void Update()
	{
	}
};

template<typename BaseType, int Size> struct e_AggregateBaseType
{
	CUDA_ALIGN(16) unsigned int type;
	//unsigned int base_type;
	CUDA_ALIGN(16) unsigned char Data[Dmax2(Dmin2(RND_16(Size), 2048), 12)];

	template<typename SpecializedType> CUDA_FUNC_IN SpecializedType* As() const
	{
		return (SpecializedType*)Data;
	}

	template<typename SpecializedType> CUDA_FUNC_IN void SetData(const SpecializedType& val)
	{
		memcpy(Data, &val, sizeof(SpecializedType));
		type = SpecializedType::TYPE();
		//base_type = SpecializedType::BASE_TYPE();
	}

	template<typename SpecializedType> CUDA_FUNC_IN bool Is() const
	{
		return type == SpecializedType::TYPE();// && base_type == SpecializedType::BASE_TYPE();
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
	CUDA_FUNC_IN e_Variable()
	{
	}
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