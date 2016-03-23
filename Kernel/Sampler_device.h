#pragma once

#include <Math/Vector.h>
#include <Base/CudaRandom.h>
#include <CudaMemoryManager.h>

#define SEQUENCE_SAMPLER

namespace CudaTracerLib {

template<typename T, int SEQUENCE_LENGTH> struct SequenceSamplerData_Sequence
{
	T data[SEQUENCE_LENGTH];
	CUDA_FUNC_IN SequenceSamplerData_Sequence() = default;
	CUDA_FUNC_IN T& operator[](unsigned int i)
	{
		CTL_ASSERT(i < SEQUENCE_LENGTH);
		return data[i];
	}
	CUDA_FUNC_IN const T& operator[](unsigned int i) const
	{
		CTL_ASSERT(i < SEQUENCE_LENGTH);
		return data[i];
	}
};

template<int N_SEQUENCES, int SEQUENCE_LENGTH> struct SequenceSampler;
template<int N_SEQUENCES, int SEQUENCE_LENGTH> struct SequenceSamplerData
{

	typedef SequenceSampler<N_SEQUENCES, SEQUENCE_LENGTH> SamplerType;

	enum { NUM_SEQUENCES = N_SEQUENCES, LEN_SEQUENCE = SEQUENCE_LENGTH };

	template<typename T> using Sequence = SequenceSamplerData_Sequence<T, SEQUENCE_LENGTH>;

private:
	Sequence<float> m_d1Data[N_SEQUENCES];
	Sequence<float2> m_d2Data[N_SEQUENCES];
	int m_num_sequences_per, m_num_sequences_per_sqrt;
public:
	CUDA_FUNC_IN SequenceSamplerData() = default;
	void Free()
	{

	}
	CUDA_FUNC_IN Sequence<float>& dim1(unsigned int i)
	{
		CTL_ASSERT(i < N_SEQUENCES);
		return m_d1Data[i];
	}
	CUDA_FUNC_IN const Sequence<float>& dim1(unsigned int i) const
	{
		CTL_ASSERT(i < N_SEQUENCES);
		return m_d1Data[i];
	}
	CUDA_FUNC_IN Sequence<float2>& dim2(unsigned int i)
	{
		CTL_ASSERT(i < N_SEQUENCES);
		return m_d2Data[i];
	}
	CUDA_FUNC_IN const Sequence<float2>& dim2(unsigned int i) const
	{
		CTL_ASSERT(i < N_SEQUENCES);
		return m_d2Data[i];
	}
	CUDA_FUNC_IN int getNumRandomsPerSequence() const
	{
		return m_num_sequences_per;
	}
	CUDA_FUNC_IN int getNumRandomsPerSequenceSqrt() const
	{
		return m_num_sequences_per_sqrt;
	}
	CUDA_FUNC_IN void setNumSequences(int val)
	{
		m_num_sequences_per = val / N_SEQUENCES;
		m_num_sequences_per_sqrt = max(1, (int)math::sqrt((float)m_num_sequences_per));
	}

	CUDA_FUNC_IN SamplerType operator()();
	CUDA_FUNC_IN void operator()(const SamplerType& val)
	{

	}
};

template<int N_SEQUENCES, int SEQUENCE_LENGTH> struct SequenceSampler
{
private:
	SequenceSamplerData<N_SEQUENCES, SEQUENCE_LENGTH>& data;
	unsigned int m_index;
	unsigned short d1, d2;
	CUDA_FUNC_IN static unsigned int wang_hash(unsigned int seed)
	{
		seed = (seed ^ 61) ^ (seed >> 16);
		seed *= 9;
		seed = seed ^ (seed >> 4);
		seed *= 0x27d4eb2d;
		seed = seed ^ (seed >> 15);
		return seed;
	}
	CUDA_FUNC_IN float map(float f) const
	{
		return fmodf(m_index / data.getNumRandomsPerSequence() + f, 1.0f) * 0.9999f;
		//return wang_hash((unsigned int)(f * UINT_MAX) + m_index / N_SEQUENCES + g_PassIndex) / float(UINT_MAX) * 0.9999f;
		//return fmodf(f + wang_hash(m_index / N_SEQUENCES + O) / float(UINT_MAX), 0.99f);
		//unsigned long long r1 = 0xca23513fefd9b3a6, r2 = 0x426b2a8687be751f;
		//char s1 = (r1 >> ((m_index / N_SEQUENCES) % 64)) & 1;
		//char s2 = (r2 >> (g_PassIndex % 64)) & 1;
		//float i = (float)(m_index / N_SEQUENCES);
		//return 0.9f * math::abs(s1 - fmod(f + i * g_SamplerData.getMappingParameter() + s2 * 0.25f, 1.0f));
	}
	CUDA_FUNC_IN Vec2f map(const Vec2f& f) const
	{
		auto n = data.getNumRandomsPerSequenceSqrt();
		int x = m_index % n, y = m_index / n;
		float u = f.x + x / float(n), v = f.y + y / float(n);
		return Vec2f(fmodf(u, 1.0f), fmodf(v, 1.0f)) * 0.9999f;
	}
public:
	CUDA_FUNC_IN SequenceSampler(unsigned int i, SequenceSamplerData<N_SEQUENCES, SEQUENCE_LENGTH>& dat)
		: m_index(i), d1(0), d2(0), data(dat)
	{

	}
	CUDA_FUNC_IN float randomFloat()
	{
		float val = data.dim1(m_index % N_SEQUENCES)[d1++];
		return map(val);
	}
	CUDA_FUNC_IN Vec2f randomFloat2()
	{
		auto val = data.dim2(m_index % N_SEQUENCES)[d2++];
		return map(Vec2f(val.x, val.y));
	}
	CUDA_FUNC_IN void StartSequence(unsigned int idx)
	{
		m_index = idx;
		d1 = d2 = 0;
	}
};

template<int N_SEQUENCES, int SEQUENCE_LENGTH> SequenceSampler<N_SEQUENCES, SEQUENCE_LENGTH> SequenceSamplerData<N_SEQUENCES, SEQUENCE_LENGTH>::operator()()
{
	return SamplerType(getGlobalIdx_2D_2D(), *this);
}

struct RandomSampler : public CudaRNG
{
	CUDA_FUNC_IN RandomSampler() = default;
	CUDA_FUNC_IN RandomSampler(unsigned int seed)
		: CudaRNG(seed)
	{
		
	}
	CUDA_FUNC_IN void StartSequence(unsigned int idx)
	{
		
	}
};

struct RandomSamplerData
{
	typedef RandomSampler SamplerType;
private:
	unsigned int m_uNumGenerators;
	RandomSampler* m_pHostGenerators;
	RandomSampler* m_pDeviceGenerators;
public:
	CUDA_FUNC_IN RandomSamplerData() = default;
	void ConstructData(unsigned int n = 1 << 14)
	{
		m_uNumGenerators = n;
		CUDA_MALLOC(&m_pDeviceGenerators, m_uNumGenerators * sizeof(RandomSampler));
		m_pHostGenerators = new RandomSampler[m_uNumGenerators];
		for (unsigned int i = 0; i < m_uNumGenerators; i++)
		{
			m_pHostGenerators[i] = RandomSampler(i);
		}
		CUDA_MEMCPY_TO_DEVICE(m_pDeviceGenerators, m_pHostGenerators, sizeof(RandomSampler) * m_uNumGenerators);
	}
	void Free()
	{
		CUDA_FREE(m_pDeviceGenerators);
		m_pDeviceGenerators = 0;
		delete[] m_pHostGenerators;
		m_pHostGenerators = 0;
	}
	CUDA_FUNC_IN void setNumSequences(int val)
	{
		
	}
	CUDA_FUNC_IN RandomSampler operator()() const
	{
		unsigned int idx = getGlobalIdx_2D_2D();
		unsigned int i = idx % m_uNumGenerators;
#ifdef ISCUDA
		RandomSampler rng = m_pDeviceGenerators[i];
		if (idx >= m_uNumGenerators)
		{
			//skipahead_sequence(idx - m_uNumGenerators, &rng.state);
		}
#else
		RandomSampler rng = m_pHostGenerators[i];
#endif
		return rng;
	}
	CUDA_FUNC_IN void operator()(const RandomSampler& val)
	{
		unsigned int i = getGlobalIdx_2D_2D();
#ifdef ISCUDA
		if (i < m_uNumGenerators)
			m_pDeviceGenerators[i] = val;
#else
		m_pHostGenerators[i % m_uNumGenerators] = val;
#endif
	}
};

typedef RandomSamplerData SamplerData;
//typedef SequenceSamplerData<5, 50> SamplerData;

typedef SamplerData::SamplerType Sampler;

class ISamplingSequenceGenerator
{
public:
	virtual ~ISamplingSequenceGenerator()
	{

	}
	virtual void Compute(float* sequence, unsigned int n) = 0;
	virtual void Compute(Vec2f* sequence, unsigned int n) = 0;
	template<int N_SEQUENCES, int SEQ_LEN> void Compute(SequenceSamplerData<N_SEQUENCES, SEQ_LEN>& data)
	{
		for (int i = 0; i < N_SEQUENCES; i++)
		{
			Compute(&data.dim1(i)[0], SEQ_LEN);
			Compute((Vec2f*)&data.dim2(i)[0], SEQ_LEN);
		}
	}
	void Compute(RandomSamplerData& data)
	{
		
	}
};

}