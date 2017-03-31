#include "CudaRandom.h"
#include <Math/Vector.h>
#include <Base/CudaMemoryManager.h>

namespace CudaTracerLib {

float Curand_GENERATOR::randomFloat()
{
	float f;
#ifdef ISCUDA
	f = curand_uniform(&state);
#else
	f = curand_uniform2(curand2(&state));
#endif
	return f * (1 - 1e-5f);//curand_uniform := (0, 1] -> [0, 1)
}

unsigned long Curand_GENERATOR::randomUint()
{
#ifdef ISCUDA
	return curand(&state);
#else
	return curand2(&state);
#endif
}

void Curand_GENERATOR::Initialize(unsigned int a_Index)
{
#ifdef ISCUDA
	curand_init(1234, a_Index, 0, &state);
#else
	curand_init2(1234, a_Index, 0, &state);
#endif
}

}