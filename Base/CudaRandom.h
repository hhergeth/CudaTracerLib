#pragma once

#include "..\Defines.h"

#include "curand_kernel.h"
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
#ifdef ISCUDA
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
private:
	unsigned int curand(curandStateXORWOW_t *state)
	{
		unsigned int t;
		t = (state->v[0] ^ (state->v[0] >> 2));
		state->v[0] = state->v[1];
		state->v[1] = state->v[2];
		state->v[2] = state->v[3];
		state->v[3] = state->v[4];
		state->v[4] = (state->v[4] ^ (state->v[4] <<4)) ^ (t ^ (t << 1));
		state->d += 362437;
		return state->v[4] + state->d;
	}
	float _curand_uniform(unsigned int x)
	{
		return x * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
	}
	void __curand_matvec(unsigned int *vector, unsigned int *matrix, 
                                unsigned int *result, int n)
	{
		for(int i = 0; i < n; i++) {
			result[i] = 0;
		}
		for(int i = 0; i < n; i++) {
			for(int j = 0; j < 32; j++) {
				if(vector[i] & (1 << j)) {
					for(int k = 0; k < n; k++) {
						result[k] ^= matrix[n * (i * 32 + j) + k];
					}
				}
			}
		}
	}
	void __curand_veccopy(unsigned int *vector, unsigned int *vectorA, int n)
	{
		for(int i = 0; i < n; i++) {
			vector[i] = vectorA[i];
		}
	}
	void __curand_matcopy(unsigned int *matrix, unsigned int *matrixA, int n)
	{
		for(int i = 0; i < n * n * 32; i++) {
			matrix[i] = matrixA[i];
		}
	}
	void __curand_matmat(unsigned int *matrixA, unsigned int *matrixB, int n)
	{
		unsigned int result[MAX_XOR_N];
		for(int i = 0; i < n * 32; i++) {
			__curand_matvec(matrixA + i * n, matrixB, result, n);
			for(int j = 0; j < n; j++) {
				matrixA[i * n + j] = result[j];
			}
		}
	}
	template <typename T, int n> void _skipahead_sequence_scratch(unsigned long long x, T *state, unsigned int *scratch)
	{
		// unsigned int matrix[n * n * 32];
		unsigned int *matrix = scratch;
		// unsigned int matrixA[n * n * 32];
		unsigned int *matrixA = scratch + (n * n * 32);
		// unsigned int vector[n];
		unsigned int *vector = scratch + (n * n * 32) + (n * n * 32);
		// unsigned int result[n];
		unsigned int *result = scratch + (n * n * 32) + (n * n * 32) + n;
		unsigned long long p = x;
		for(int i = 0; i < n; i++) {
			vector[i] = state->v[i];
		}
		int matrix_num = 0;
		while(p && matrix_num < PRECALC_NUM_MATRICES - 1) {
			for(unsigned int t = 0; t < (p & PRECALC_BLOCK_MASK); t++) {
				__curand_matvec(vector, precalc_xorwow_matrix_host[matrix_num], result, n);
				__curand_veccopy(vector, result, n);
			}
			p >>= PRECALC_BLOCK_SIZE;
			matrix_num++;
		}
		if(p) {
			__curand_matcopy(matrix, precalc_xorwow_matrix_host[PRECALC_NUM_MATRICES - 1], n);
			__curand_matcopy(matrixA, precalc_xorwow_matrix_host[PRECALC_NUM_MATRICES - 1], n);
		}
		while(p) {
			for(unsigned int t = 0; t < (p & SKIPAHEAD_MASK); t++) {
				__curand_matvec(vector, matrixA, result, n);
				__curand_veccopy(vector, result, n);
			}
			p >>= SKIPAHEAD_BLOCKSIZE;
			if(p) {
				for(int i = 0; i < SKIPAHEAD_BLOCKSIZE; i++) {
					__curand_matmat(matrix, matrixA, n);
					__curand_matcopy(matrixA, matrix, n);
				}
			}
		}
		for(int i = 0; i < n; i++) {
			state->v[i] = vector[i];
		}
		/* No update of state->d needed, guaranteed to be a multiple of 2^32 */
	}
	template <typename T, int n> void _skipahead_scratch(unsigned long long x, T *state, unsigned int *scratch)
	{
		// unsigned int matrix[n * n * 32];
		unsigned int *matrix = scratch;
		// unsigned int matrixA[n * n * 32];
		unsigned int *matrixA = scratch + (n * n * 32);
		// unsigned int vector[n];
		unsigned int *vector = scratch + (n * n * 32) + (n * n * 32);
		// unsigned int result[n];
		unsigned int *result = scratch + (n * n * 32) + (n * n * 32) + n;
		unsigned long long p = x;
		for(int i = 0; i < n; i++) {
			vector[i] = state->v[i];
		}
		int matrix_num = 0;
		while(p && (matrix_num < PRECALC_NUM_MATRICES - 1)) {
			for(unsigned int t = 0; t < (p & PRECALC_BLOCK_MASK); t++) {
				__curand_matvec(vector, precalc_xorwow_offset_matrix_host[matrix_num], result, n);
				__curand_veccopy(vector, result, n);
			}
			p >>= PRECALC_BLOCK_SIZE;
			matrix_num++;
		}
		if(p) {
			__curand_matcopy(matrix, precalc_xorwow_offset_matrix_host[PRECALC_NUM_MATRICES - 1], n);
			__curand_matcopy(matrixA, precalc_xorwow_offset_matrix_host[PRECALC_NUM_MATRICES - 1], n);
		}
		while(p) {
			for(unsigned int t = 0; t < (p & SKIPAHEAD_MASK); t++) {
				__curand_matvec(vector, matrixA, result, n);
				__curand_veccopy(vector, result, n);
			}
			p >>= SKIPAHEAD_BLOCKSIZE;
			if(p) {
				for(int i = 0; i < SKIPAHEAD_BLOCKSIZE; i++) {
					__curand_matmat(matrix, matrixA, n);
					__curand_matcopy(matrixA, matrix, n);
				}
			}
		}
		for(int i = 0; i < n; i++) {
			state->v[i] = vector[i];
		}
		state->d += 362437 * (unsigned int)x;
	}
	void _curand_init_scratch(unsigned long long seed, 
										 unsigned long long subsequence, 
										 unsigned long long offset, 
										 curandStateXORWOW_t *state,
										 unsigned int *scratch)
	{
		unsigned int s0 = ((unsigned int)seed) ^ 0xaad26b49UL;
		unsigned int s1 = (unsigned int)(seed >> 32) ^ 0xf7dcefddUL;
		unsigned int t0 = 1099087573UL * s0;
		unsigned int t1 = 2591861531UL * s1;
		state->d = 6615241 + t1 + t0;
		state->v[0] = 123456789UL + t0;
		state->v[1] = 362436069UL ^ t0;
		state->v[2] = 521288629UL + t1;
		state->v[3] = 88675123UL ^ t1;
		state->v[4] = 5783321UL + t0;
		_skipahead_sequence_scratch<curandStateXORWOW_t, 5>(subsequence, state, scratch);
		_skipahead_scratch<curandStateXORWOW_t, 5>(offset, state, scratch);
		state->boxmuller_flag = 0;
		state->boxmuller_flag_double = 0;
	}
	void curand_init(unsigned long long seed, 
                            unsigned long long subsequence, 
                            unsigned long long offset, 
                            curandStateXORWOW_t *state)
	{
		unsigned int scratch[5 * 5 * 32 * 2 + 5 * 2];
		_curand_init_scratch(seed, subsequence, offset, state, (unsigned int*)scratch);
	}
public:
	CUDA_FUNC_IN void Initialize(unsigned int a_Index, unsigned int a_Spacing, unsigned int a_Offset)
	{
		curand_init(a_Index * a_Spacing, a_Index * a_Offset, 0, &state);
	}
	CUDA_FUNC_IN float randomFloat()
	{
		return _curand_uniform(curand(&state));
	}
	CUDA_FUNC_IN unsigned long randomUint()
	{
		return curand(&state);
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
	curandState* m_pHostGenerators;
	curandState* m_pDeviceGenerators;
public:
	unsigned int m_uOffset;
public:
	//curandSetGeneratorOffset(GENERATOR[i], i * a_Offset);
	k_TracerRNGBuffer(unsigned int a_Length, unsigned int a_Spacing = 1234, unsigned int a_Offset = 0)
	{
		m_uOffset = 0;
		m_uNumGenerators = a_Length;
		cudaMalloc(&m_pDeviceGenerators, a_Length * sizeof(curandState));
		m_pHostGenerators = new curandState[a_Length];
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
		unsigned int i = threadId_Unsafe % m_uNumGenerators;
#ifdef __CUDACC__
		r.state = m_pDeviceGenerators[i];
#else
		r.state = m_pHostGenerators[i];
#endif
		return r;
	}
	CUDA_FUNC_IN void operator()(k_TracerRNG& val)
	{
#ifdef __CUDACC__
		unsigned int i = threadId;
		if(i < m_uNumGenerators)
			m_pDeviceGenerators[i] = val.state;
#else
		m_pHostGenerators[threadId_Unsafe % m_uNumGenerators] = val.state;
#endif
	}
private:
	void createGenerators(unsigned int a_Spacing, unsigned int a_Offset);
};

typedef k_TracerRNG CudaRNG;