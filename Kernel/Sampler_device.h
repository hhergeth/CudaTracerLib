#pragma once

#include <Math/Vector.h>
#include <Base/CudaRandom.h>
#include <CudaMemoryManager.h>
#include <Engine/SynchronizedBuffer.h>

namespace CudaTracerLib {

struct SequenceSampler;
struct SequenceSamplerData : public ISynchronizedBufferParent
{
	typedef SequenceSampler SamplerType;
private:
	SynchronizedBuffer<float> m_d1Data;
	SynchronizedBuffer<Vec2f> m_d2Data;
	unsigned int num_sequences;
	unsigned int sequence_length;

	CUDA_FUNC_IN unsigned element_idx(unsigned int sequence_idx, unsigned int element_idx) const
	{
		//assume sequence_idx is increasing over lanes
		return element_idx * num_sequences + sequence_idx;
	}
public:
	SequenceSamplerData(unsigned int num_sequences, unsigned int sequence_length)
		: m_d1Data(num_sequences * sequence_length), m_d2Data(num_sequences * sequence_length), ISynchronizedBufferParent(m_d1Data, m_d2Data),
		  num_sequences(num_sequences), sequence_length(sequence_length)
	{

	}

	CUDA_FUNC_IN unsigned int getNumSequences() const
	{
		return num_sequences;
	}

	CUDA_FUNC_IN unsigned int getSequenceLength() const
	{
		return sequence_length;
	}

	CUDA_FUNC_IN float& getSequenceElement1(unsigned int sequence_idx, unsigned int element_idx)
	{
		return m_d1Data[this->element_idx(sequence_idx, element_idx)];
	}

	CUDA_FUNC_IN Vec2f& getSequenceElement2(unsigned int sequence_idx, unsigned int element_idx)
	{
		return m_d2Data[this->element_idx(sequence_idx, element_idx)];
	}

	CUDA_FUNC_IN SamplerType operator()(unsigned int idx);
	CUDA_FUNC_IN void operator()(const SamplerType& val, unsigned int idx)
	{
	}
};

struct SequenceSampler
{
private:
	template<unsigned k> struct SequenceCombinerK
	{
		template<typename F> static CUDA_FUNC_IN void compute_coordinates_from_index(unsigned int flattened_idx, unsigned int length, F& clb)
		{
			for (unsigned int i = 0; i < k; i++)
			{
				clb(flattened_idx % length);
				flattened_idx = flattened_idx / length;
			}
		}

		template<typename T, typename F> static CUDA_FUNC_IN T compute_sequences_sum(unsigned int random_idx, unsigned int num_sequences, F& el_clb)
		{
			T val = T(0);
			compute_coordinates_from_index(random_idx, num_sequences, [&](unsigned int coord_i)
			{
				T el = el_clb(coord_i);
				val += el;
			});
			return val;
		}
	};

	typedef SequenceCombinerK<2> SequenceCombiner;

	SequenceSamplerData& data;
	unsigned int random_idx;
	unsigned int d1_idx, d2_idx;
public:
	CUDA_FUNC_IN SequenceSampler(unsigned int idx, SequenceSamplerData& dat)
		:  data(dat), random_idx(idx)
	{
		d1_idx = d2_idx = 0;
	}
	CUDA_FUNC_IN float randomFloat()
	{
		auto sum = SequenceCombiner::compute_sequences_sum<float>(random_idx, data.getNumSequences(), [&](unsigned int sequence_idx) {return data.getSequenceElement1(sequence_idx, d1_idx % data.getSequenceLength()); });
		d1_idx++;
		return math::frac(sum);
	}
	CUDA_FUNC_IN Vec2f randomFloat2()
	{
		auto sum = SequenceCombiner::compute_sequences_sum<Vec2f>(random_idx, data.getNumSequences(), [&](unsigned int sequence_idx) {return data.getSequenceElement2(sequence_idx, d2_idx % data.getSequenceLength()); });
		d2_idx++;
		return Vec2f(math::frac(sum.x), math::frac(sum.y));
	}
	CUDA_FUNC_IN void skip(unsigned int off)
	{
		d1_idx += off;
		d2_idx += off;
	}
};

SequenceSampler SequenceSamplerData::operator()(unsigned int idx)
{
	return SamplerType(idx, *this);
}

struct RandomSampler : public CudaRNG
{
	CUDA_FUNC_IN RandomSampler() = default;
	CUDA_FUNC_IN RandomSampler(unsigned int seed)
		: CudaRNG(seed)
	{

	}
	CUDA_FUNC_IN void skip(unsigned int off)
	{
		//not necessary
	}
};

struct RandomSamplerData : ISynchronizedBufferParent
{
	typedef RandomSampler SamplerType;
private:
	SynchronizedBuffer<RandomSampler> m_samplerBuffer;
public:
	RandomSamplerData(unsigned int num_sequences, unsigned int sequence_length)
		: m_samplerBuffer(num_sequences)
	{
		for (unsigned int i = 0; i < m_samplerBuffer.getLength(); i++)
		{
			m_samplerBuffer[i] = RandomSampler(i);
		}
		m_samplerBuffer.setOnCPU();
		m_samplerBuffer.Synchronize();
	}

	CUDA_FUNC_IN unsigned int getNumSequences() const
	{
		return m_samplerBuffer.getLength();
	}

	CUDA_FUNC_IN RandomSampler operator()(unsigned int idx) const
	{
		unsigned int i = idx % m_samplerBuffer.getLength();
		return m_samplerBuffer[i];
	}
	CUDA_FUNC_IN void operator()(const RandomSampler& val, unsigned int idx)
	{
		if (idx < m_samplerBuffer.getLength())
			m_samplerBuffer[idx] = val;
	}
};

//typedef RandomSamplerData SamplerData;
typedef SequenceSamplerData SamplerData;

typedef SamplerData::SamplerType Sampler;

class ISamplingSequenceGenerator
{
public:
	virtual ~ISamplingSequenceGenerator()
	{

	}
	virtual void Compute1D(float* sequence, unsigned int sequence_idx, unsigned int sequence_length) = 0;
	virtual void Compute2D(Vec2f* sequence, unsigned int sequence_idx, unsigned int sequence_length) = 0;
	void Compute(SequenceSamplerData& data)
	{
		float* sequence_1d = (float*)alloca(sizeof(float) * data.getSequenceLength());
		Vec2f* sequence_2d = (Vec2f*)alloca(sizeof(Vec2f) * data.getSequenceLength());
		for (unsigned int sequence_idx = 0; sequence_idx < data.getNumSequences(); sequence_idx++)
		{
			Compute1D(sequence_1d, sequence_idx, data.getSequenceLength());
			Compute2D(sequence_2d, sequence_idx, data.getSequenceLength());
			for (unsigned int i = 0; i < data.getSequenceLength(); i++)
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

}