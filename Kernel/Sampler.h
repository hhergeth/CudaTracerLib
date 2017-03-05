#pragma once

#include "Sampler_device.h"
#include "TraceHelper.h"

namespace CudaTracerLib {

class ISamplingSequenceGenerator
{
public:
	virtual ~ISamplingSequenceGenerator()
	{

	}
	virtual void Compute(SequenceSamplerData& data) = 0;
	virtual void Compute(RandomSamplerData& data)
	{

	}
};

template<typename Driver> class SamplingSequenceGeneratorHost : public ISamplingSequenceGenerator
{
	Driver _obj;
public:
	SamplingSequenceGeneratorHost()
	{

	}

	SamplingSequenceGeneratorHost(const Driver& obj)
		: _obj(obj)
	{
	}

	virtual void Compute(SequenceSamplerData& data)
	{
		auto seqLen = data.getSequenceLength();
		float* sequence_1d = (float*)alloca(sizeof(float) * seqLen);
		Vec2f* sequence_2d = (Vec2f*)alloca(sizeof(Vec2f) * seqLen);
		for(unsigned int sequence_idx = 0; sequence_idx < data.getNumSequences(); sequence_idx++)
		{
			_obj.Compute1D(sequence_1d, sequence_idx, seqLen);
			_obj.Compute2D(sequence_2d, sequence_idx, seqLen);
			for (unsigned int i = 0; i < seqLen; i++)
			{
				data.getSequenceElement1(sequence_idx, i) = sequence_1d[i];
				data.getSequenceElement2(sequence_idx, i) = sequence_2d[i];
			}
		}

		data.setOnCPU();
		data.Synchronize();
	}

	virtual void Compute(RandomSamplerData& data)
	{

	}
};

class IndependantSamplingSequenceGenerator
{
	CudaRNG rng;
public:
	IndependantSamplingSequenceGenerator()
		: rng(7539414)
	{
	}
	void Compute1D(float* sequence, unsigned int sequence_idx, unsigned int sequence_length)
	{
		for (unsigned int i = 0; i < sequence_length; i++)
			sequence[i] = rng.randomFloat();
	}
	void Compute2D(Vec2f* sequence, unsigned int sequence_idx, unsigned int sequence_length)
	{
		for (unsigned int i = 0; i < sequence_length; i++)
			sequence[i] = Vec2f(rng.randomFloat(), rng.randomFloat());
	}
};

class StratifiedSamplingSequenceGenerator
{
	const int n_strata;
public:
	StratifiedSamplingSequenceGenerator(int n_strata = 10)
		: n_strata(n_strata)
	{
	}
	void Compute1D(float* sequence, unsigned int sequence_idx, unsigned int sequence_length)
	{

	}
	void Compute2D(Vec2f* sequence, unsigned int sequence_idx, unsigned int sequence_length)
	{

	}
};

}