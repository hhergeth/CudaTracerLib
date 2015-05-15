#pragma once

#include "../Math/Vector.h"
#include "curand_kernel.h"

class LinearCongruental_GENERATOR
{
	unsigned int X_i;
public:
	CUDA_FUNC_IN LinearCongruental_GENERATOR(unsigned int seed = 12345)
	{
		X_i = seed;
	}

	CUDA_FUNC_IN unsigned int randomUint()
	{
		unsigned int a = 1664525, c = 1013904223;
		X_i = (a * X_i + c);
		return X_i;
	}

	CUDA_FUNC_IN float randomFloat()
	{
		return float(randomUint()) / float(0xffffffff);
	}
};

class Lehmer_GENERATOR
{
	unsigned int X_i;
public:
	CUDA_FUNC_IN Lehmer_GENERATOR(unsigned int seed = 123456789L)
	{
		X_i = seed;
	}

	CUDA_FUNC_IN unsigned int randomUint()
	{
		const unsigned int a = 16807;
		const unsigned int m = 2147483647;
		X_i = (unsigned int(X_i * a)) % m;
		return X_i;
	}

	CUDA_FUNC_IN float randomFloat()
	{
		return randomUint() / float(2147483647);
	}
};

class TAUSWORTHE_GENERATOR
{
	CUDA_FUNC_IN unsigned int TausStep(unsigned int &z, unsigned int S1, unsigned int S2, unsigned int S3, unsigned int M)
	{
		unsigned int b=(((z << S1) ^ z) >> S2); 
		return z = (((z & M) << S3) ^ b); 
	}
	CUDA_FUNC_IN unsigned int LCGStep(unsigned int &z, unsigned int A, unsigned int C)
	{
		return z=(A*z+C);
	}
	unsigned int z1, z2, z3, z4;
public:
	CUDA_FUNC_IN TAUSWORTHE_GENERATOR()
	{

	}

	CUDA_FUNC_IN TAUSWORTHE_GENERATOR(unsigned int seed)
	{
		z1 = seed + 1 + 1;
		z2 = seed + 7 + 1;
		z3 = seed + 15 + 1;
		z4 = seed + 127 + 1;
	}

	CUDA_FUNC_IN unsigned int randomUint()
	{
		return TausStep(z1, 13, 19, 12, 4294967294UL) ^ TausStep(z2, 2, 25, 4, 4294967288UL) ^ TausStep(z3, 3, 11, 17, 4294967280UL) ^ LCGStep(z4, 1664525, 1013904223UL);
	}

	CUDA_FUNC_IN float randomFloat()
	{
		return float(randomUint()) / float(0xffffffff);
	}
};

class Xorshift_GENERATOR
{
	unsigned int y;
public:
	CUDA_FUNC_IN Xorshift_GENERATOR()
	{
#ifndef ISCUDA
		y = 123456789;
#endif
	}

	CUDA_FUNC_IN Xorshift_GENERATOR(unsigned int seed)
	{
		y = seed;
	}

	CUDA_FUNC_IN unsigned int randomUint()
	{
		y = y ^ (y << 13);
		y = y ^ (y >> 17);
		y = y ^ (y << 5);
		return y;
	}

	CUDA_FUNC_IN float randomFloat()
	{
		return float(randomUint()) / float(0xffffffff);
	}
};

struct k_TracerRNG_cuRAND
{
	curandState state;
private:
	unsigned int curand2(curandStateXORWOW_t *state)
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
	float curand_uniform2(unsigned int x)
	{
		return x * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
	}
	void __curand_matvec(unsigned int *vector, unsigned int *matrix, unsigned int *result, int n)
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
	void curand_init2(unsigned long long seed, 
                            unsigned long long subsequence, 
                            unsigned long long offset, 
                            curandStateXORWOW_t *state)
	{
		unsigned int scratch[5 * 5 * 32 * 2 + 5 * 2];
		_curand_init_scratch(seed, subsequence, offset, state, (unsigned int*)scratch);
	}
public:
	CUDA_DEVICE CUDA_HOST void Initialize(unsigned int a_Index, unsigned int a_Spacing, unsigned int a_Offset);
	CUDA_DEVICE CUDA_HOST float randomFloat();
	CUDA_DEVICE CUDA_HOST unsigned long randomUint();

	CUDA_FUNC_IN Vec2f randomFloat2()
	{
		return Vec2f(randomFloat(), randomFloat());
	}
	CUDA_FUNC_IN Vec3f randomFloat3()
	{
		return Vec3f(randomFloat(), randomFloat(), randomFloat());
	}
	CUDA_FUNC_IN Vec4f randomFloat4()
	{
		return Vec4f(randomFloat(), randomFloat(), randomFloat(), randomFloat());
	}
};

class CudaRNGBuffer_cuRAND
{
private:
	unsigned int m_uNumGenerators;
	k_TracerRNG_cuRAND* m_pHostGenerators;
	k_TracerRNG_cuRAND* m_pDeviceGenerators;
public:
	//curandSetGeneratorOffset(GENERATOR[i], i * a_Offset);
	CudaRNGBuffer_cuRAND(unsigned int a_Length, unsigned int a_Spacing = 1234, unsigned int a_Offset = 0);
	CudaRNGBuffer_cuRAND(){}
	void Free();
	CUDA_DEVICE CUDA_HOST k_TracerRNG_cuRAND operator()();
	CUDA_DEVICE CUDA_HOST void operator()(k_TracerRNG_cuRAND& val);
	void NextPass();
private:
	void createGenerators(unsigned int a_Spacing, unsigned int a_Offset);
};

typedef k_TracerRNG_cuRAND CudaRNG;
typedef CudaRNGBuffer_cuRAND CudaRNGBuffer;