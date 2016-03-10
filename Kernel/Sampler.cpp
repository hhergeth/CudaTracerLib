#include <StdAfx.h>
#include "Sampler.h"

namespace CudaTracerLib {

SamplerData ConstructDefaultSamplerData()
{
#ifdef SEQUENCE_SAMPLER
	return SamplerData(1.5f);
#else
	return SamplerData(15000);
#endif
}

}
