#pragma once

#include "Sampler_device.h"
#include "TraceHelper.h"

namespace CudaTracerLib {

class StratifiedSequenceGenerator : public ISamplingSequenceGenerator
{
	const int n_strata;
public:
	StratifiedSequenceGenerator(int n_strata = 10)
		: n_strata(n_strata)
	{
	}
	virtual void Compute(float* sequence, unsigned int n)
	{

	}
	virtual void Compute(Vec2f* sequence, unsigned int n)
	{

	}
};

}