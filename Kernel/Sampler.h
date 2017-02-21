#pragma once

#include "Sampler_device.h"
#include "TraceHelper.h"

namespace CudaTracerLib {

class IndependantSamplingSequenceGenerator : public ISamplingSequenceGenerator
{
	CudaRNG rng;
public:
	IndependantSamplingSequenceGenerator()
		: rng(1237543)
	{
	}
	virtual void Compute1D(float* sequence, unsigned int sequence_idx, unsigned int sequence_length)
	{
		for (unsigned int i = 0; i < sequence_length; i++)
			sequence[i] = rng.randomFloat();
	}
	virtual void Compute2D(Vec2f* sequence, unsigned int sequence_idx, unsigned int sequence_length)
	{
		for (unsigned int i = 0; i < sequence_length; i++)
			sequence[i] = rng.randomFloat2();
	}
};

class StratifiedSamplingSequenceGenerator : public ISamplingSequenceGenerator
{
	const int n_strata;
public:
	StratifiedSamplingSequenceGenerator(int n_strata = 10)
		: n_strata(n_strata)
	{
	}
	virtual void Compute1D(float* sequence, unsigned int sequence_idx, unsigned int sequence_length)
	{

	}
	virtual void Compute2D(Vec2f* sequence, unsigned int sequence_idx, unsigned int sequence_length)
	{

	}
};

}