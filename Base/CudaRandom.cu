#include "CudaRandom.h"

#include "..\MathTypes.h"

float k_TracerRNG_cuRAND::randomFloat()
{
#ifdef ISCUDA
	return curand_uniform(&state);
#else
	return curand_uniform2(curand2(&state));
#endif
}

unsigned long k_TracerRNG_cuRAND::randomUint()
{
#ifdef ISCUDA
	return curand(&state);
#else
	return curand2(&state);
#endif
}

void k_TracerRNG_cuRAND::Initialize(unsigned int a_Index, unsigned int a_Spacing, unsigned int a_Offset)
{
#ifdef ISCUDA
	curand_init(a_Index * a_Spacing, a_Index * a_Offset, 0, &state);
#else
	curand_init2(a_Index * a_Spacing, a_Index * a_Offset, 0, &state);
#endif
}

CudaRNGBuffer_cuRAND::CudaRNGBuffer_cuRAND(unsigned int a_Length, unsigned int a_Spacing, unsigned int a_Offset)
{
	m_uNumGenerators = a_Length;
	cudaMalloc(&m_pDeviceGenerators, a_Length * sizeof(k_TracerRNG_cuRAND));
	m_pHostGenerators = new k_TracerRNG_cuRAND[a_Length];
	createGenerators(a_Spacing, a_Offset);
}

void CudaRNGBuffer_cuRAND::createGenerators(unsigned int a_Spacing, unsigned int a_Offset)
{
	for(unsigned int i = 0; i < m_uNumGenerators; i++)
	{
		(m_pHostGenerators + i)->Initialize(i, a_Spacing, a_Offset);
	}
	cudaMemcpy(m_pDeviceGenerators, m_pHostGenerators, sizeof(k_TracerRNG_cuRAND) * m_uNumGenerators, cudaMemcpyHostToDevice);
}

k_TracerRNG_cuRAND CudaRNGBuffer_cuRAND::operator()()
{
	unsigned int i = threadId % m_uNumGenerators;
#ifdef ISCUDA
	return m_pDeviceGenerators[i];
#else
	return m_pHostGenerators[i];
#endif
}

void CudaRNGBuffer_cuRAND::operator()(k_TracerRNG_cuRAND& val)
{
#ifdef ISCUDA
	unsigned int i = threadId;
	if(i < m_uNumGenerators)
		m_pDeviceGenerators[i] = val;
#else
	m_pHostGenerators[threadId % m_uNumGenerators] = val;
#endif
}

void CudaRNGBuffer_cuRAND::NextPass()
{

}

float k_Tracer_sobol::randomFloat()
{
	return state.randomFloat();
}

unsigned long k_Tracer_sobol::randomUint()
{
	return state.randomUint();
}

k_Tracer_sobol_Buffer::k_Tracer_sobol_Buffer(unsigned int a_Length, unsigned int a_Spacing, unsigned int a_Offset)
{
	passIndex = 0;
}

void k_Tracer_sobol_Buffer::NextPass()
{
	passIndex++;
	int N = passIndex;

	unsigned int s[32] = { 0,  1,  2,  3,  3,  4,  4,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7 };
	unsigned int a[32] = { 0,  0,  1,  1,  2,  1,  4,  2, 13,  7, 14, 11,  4,  1, 16, 13, 22, 19, 25,  1, 32,  4,  8,  7, 56, 14, 28, 19, 50, 21, 42, 31 }; 
    unsigned int m[32][8] = { { 0, 1, 0, 0, 0, 0, 0, 0 }, { 0, 1, 1, 0, 0, 0, 0, 0 }, { 0, 1, 1, 1, 0, 0, 0, 0 }, { 0, 1, 3, 1, 0, 0, 0, 0 }, 
                        { 0, 1, 1, 7, 13, 0, 0, 0 }, { 0, 1, 1, 3, 7, 0, 0, 0 }, { 0, 1, 3, 1, 7, 21, 0, 0 }, { 0, 1, 3, 1, 3, 9, 0, 0 }, 
                        { 0, 1, 1, 5, 9, 13, 0, 0 }, { 0, 1, 1, 3, 9, 13, 0, 0 }, { 0, 1, 1, 5, 3, 7, 0, 0 }, { 0, 1, 1, 5, 7, 11, 0, 0 },
                        { 0, 1, 3, 3, 13, 15, 43, 0 }, { 0, 1, 3, 5, 11, 25, 45, 0 }, { 0, 1, 1, 3, 11, 3, 45, 0 }, { 0, 1, 3, 5, 1, 9, 21, 0 }, 
                        { 0, 1, 1, 3, 13, 9, 9, 0 }, { 0, 1, 3, 5, 7, 17, 53, 0 }, { 0, 1, 3, 1, 7, 11, 51, 115 }, { 0, 1, 1, 7, 9, 25, 35, 11 }, 
                        { 0, 1, 3, 1, 1, 31, 5, 1 }, { 0, 1, 1, 5, 9, 11, 1, 121 }, { 0, 1, 1, 1, 15, 11, 59, 21 }, { 0, 1, 3, 1, 3, 17, 49, 51 },
                        { 0, 1, 1, 5, 7, 15, 25, 13 }, { 0, 1, 3, 7, 7, 1, 45, 7 }, { 0, 1, 1, 5, 7, 21, 21, 37 }, { 0, 1, 3, 1, 13, 9, 49, 23 }, 
                        { 0, 1, 1, 1, 3, 3, 35, 123 }, { 0, 1, 1, 7, 5, 15, 47, 117 }, { 0, 1, 1, 3, 13, 9, 23, 33 } };
    unsigned int V[33], C = 1, value = N;
    const float rpf = 1.0f / powf( 2.0f, 32 );
    while (value & 1) value >>= 1, C++;
    for (unsigned int i = 1; i <= 32; i++ )
		V[i] = 1 << (32 - i); // all m's = 1
    X[0] = X[0] ^ V[C], points[0] = X[0] * rpf;
    for ( unsigned int j = 1; j <= 31; X[j] = X[j] ^ V[C], points[j] = X[j] * rpf, j++ ) 
        if (32 <= s[j])
			for( unsigned int i = 1; i <= 32; i++ )
				V[i] = m[j][i] << (32 - i);
		else 
        {
			for( unsigned int i = 1; i <= s[j]; i++ )
				V[i] = m[j][i] << (32 - i); 
			for( unsigned int i = s[j] + 1; i <= 32; i++ ) 
			{
				V[i] = V[i-s[j]] ^ (V[i - s[j]] >> s[j]); 
				for( unsigned int k = 1; k <= s[j] - 1; k++ )
					V[i] ^= (((a[j] >> (s[j] - 1 - k)) & 1) * V[i - k]); 
			}
        }
}

k_Tracer_sobol k_Tracer_sobol_Buffer::operator()()
{
	k_Tracer_sobol r;
	r.state = GENERATOR(threadId + passIndex*12345);
	return r;
}

void k_Tracer_sobol_Buffer::operator()(k_Tracer_sobol& val)
{

}