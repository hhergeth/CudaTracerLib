#pragma once

#include "..\Defines.h"

#include "curand_kernel.h"
/*
class CudaRNG
{
public:
	curandState state;
public:
	CUDA_FUNC_IN CudaRNG()
	{

	}

	CUDA_ONLY_FUNC CudaRNG(unsigned long long seed)
	{
		curand_init ( seed, 0, 0, &state );
	}

	CUDA_ONLY_FUNC CudaRNG(unsigned long long seed, unsigned long long sub)
	{
		curand_init ( seed, sub, 0, &state );
	}

	CUDA_ONLY_FUNC unsigned long randomUint()
	{
		return curand(&state);
	}

	CUDA_ONLY_FUNC float randomFloat()
	{
		return curand_uniform( &state );
	}
};
*/
/*
class CudaRNG
{
	struct Seed
	{
		unsigned int s1, s2, s3;
	};
private:
	Seed s;
private:
	#define FLOATMASK 0x00ffffffu

	CUDA_FUNC_IN unsigned int TAUSWORTHE(const unsigned int s, const unsigned int a,	const unsigned int b, const unsigned int c,	const unsigned int d)
	{
		return ((s&c)<<d) ^ (((s << a) ^ s) >> b);
	}
	CUDA_FUNC_IN unsigned int LCG(const unsigned int x) { return x * 69069; }

	CUDA_FUNC_IN unsigned int ValidSeed(const unsigned int x, const unsigned int m) { return (x < m) ? (x + m) : x; }
public:
	CUDA_FUNC_IN CudaRNG()
	{

	}

	CUDA_FUNC_IN CudaRNG(unsigned int seed)
	{
		// Avoid 0 value
		seed = (seed == 0) ? (seed + 0xffffffu) : seed;

		s.s1 = ValidSeed(LCG(seed), 1);
		s.s2 = ValidSeed(LCG(s.s1), 7);
		s.s3 = ValidSeed(LCG(s.s2), 15);
	}

	CUDA_FUNC_IN unsigned long randomUint()
	{
		s.s1 = TAUSWORTHE(s.s1, 13, 19, 4294967294UL, 12);
		s.s2 = TAUSWORTHE(s.s2, 2, 25, 4294967288UL, 4);
		s.s3 = TAUSWORTHE(s.s3, 3, 11, 4294967280UL, 17);

		return ((s.s1) ^ (s.s2) ^ (s.s3));
	}

	CUDA_FUNC_IN float randomFloat()
	{
		return (randomUint() & FLOATMASK) * (1.f / (FLOATMASK + 1UL));
	}
};
*/

struct k_TracerRNG
{
	friend class k_TracerRNGBuffer;
private:
	curandState state;
public:
#ifdef __CUDA_ARCH__
	CUDA_ONLY_FUNC void Initialize(unsigned int a_Index, unsigned int a_Spacing, unsigned int a_Offset)
	{
		curand_init(a_Index * a_Spacing, a_Index * a_Offset, 0, &state);
	}
	CUDA_ONLY_FUNC float randomFloat()
	{
		return curand_uniform(&state);
	}
	CUDA_ONLY_FUNC unsigned long randomUint()
	{
		return curand(&state);
	}
#else
	CUDA_FUNC_IN void Initialize(unsigned int a_Index, unsigned int a_Spacing, unsigned int a_Offset)
	{

	}
	CUDA_FUNC_IN float randomFloat()
	{
		return 0;
	}
	CUDA_FUNC_IN unsigned long randomUint()
	{
		return 0;
	}
#endif
	CUDA_FUNC_IN float2 randomFloat2()
	{
		return make_float2(randomFloat(), randomFloat());
	}
	CUDA_FUNC_IN float3 randomFloat3()
	{
		return make_float3(randomFloat(), randomFloat(), randomFloat());
	}
	CUDA_FUNC_IN float4 randomFloat4()
	{
		return make_float4(randomFloat(), randomFloat(), randomFloat(), randomFloat());
	}
};

class k_TracerRNGBuffer
{
private:
	unsigned int m_uNumGenerators;
	curandState* m_pGenerators;
public:
	unsigned int m_uOffset;
public:
	//curandSetGeneratorOffset(GENERATOR[i], i * a_Offset);
	k_TracerRNGBuffer(unsigned int a_Length, unsigned int a_Spacing = 1234, unsigned int a_Offset = 0)
	{
		m_uOffset = 0;
		m_uNumGenerators = a_Length;
		cudaMalloc(&m_pGenerators, a_Length * sizeof(curandState));
		createGenerators(a_Spacing, a_Offset);
	}
	k_TracerRNGBuffer(){}
	~k_TracerRNGBuffer()
	{
		//cudaFree(m_pGenerators);
	}
	CUDA_FUNC_IN k_TracerRNG operator()()
	{
		k_TracerRNG r;
		unsigned int i = threadId % m_uNumGenerators;
		r.state = m_pGenerators[i];
		return r;
	}
	CUDA_FUNC_IN void operator()(k_TracerRNG& val)
	{
		unsigned int i = threadId;
		if(i < m_uNumGenerators)
			m_pGenerators[i] = val.state;
	}
private:
	void createGenerators(unsigned int a_Spacing, unsigned int a_Offset);
};

typedef k_TracerRNG CudaRNG;