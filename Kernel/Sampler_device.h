#pragma once

#include <Math/Vector.h>
#include <Base/CudaRandom.h>
#include <CudaMemoryManager.h>

#define SEQUENCE_SAMPLER

namespace CudaTracerLib {

struct SequenceSampler;
struct SequenceSamplerData
{
	typedef SequenceSampler SamplerType;
private:
	float* m_d1DataDevice;
	float* m_d1DataHost;
	Vec2f* m_d2DataDevice;
	Vec2f* m_d2DataHost;
	unsigned int m_numSequences;
	int m_num_sequences_per, m_num_sequences_per_sqrt;
	int m_passIdx;
public:
	CUDA_FUNC_IN SequenceSamplerData() = default;
	void Free()
	{
		CUDA_FREE(m_d1DataDevice); m_d1DataDevice = 0;
		CUDA_FREE(m_d2DataDevice); m_d2DataDevice = 0;
		delete[] m_d1DataHost; m_d1DataHost = 0;
		delete[] m_d2DataHost; m_d2DataHost = 0;
	}
	void ConstructData(unsigned int N = 1 << 14)
	{
		m_numSequences = N;
		CUDA_MALLOC(&m_d1DataDevice, m_numSequences * sizeof(float));
		CUDA_MALLOC(&m_d2DataDevice, m_numSequences * sizeof(Vec2f));
		m_d1DataHost = new float[m_numSequences];
		m_d2DataHost = new Vec2f[m_numSequences];
	}
	void CopyToDevice()
	{
		CUDA_MEMCPY_TO_DEVICE(m_d1DataDevice, m_d1DataHost, m_numSequences * sizeof(float));
		CUDA_MEMCPY_TO_DEVICE(m_d2DataDevice, m_d2DataHost, m_numSequences * sizeof(Vec2f));
	}
	CUDA_FUNC_IN float& dim1(unsigned int i)
	{
		CTL_ASSERT(i < m_numSequences);
#ifdef ISCUDA
		return m_d1DataDevice[i];
#else
		return m_d1DataHost[i];
#endif
	}
	CUDA_FUNC_IN Vec2f& dim2(unsigned int i)
	{
		CTL_ASSERT(i < m_numSequences);
#ifdef ISCUDA
		return m_d2DataDevice[i];
#else
		return m_d2DataHost[i];
#endif
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
		m_num_sequences_per = max(1u, val / m_numSequences);
		m_num_sequences_per_sqrt = max(1, (int)math::sqrt((float)m_num_sequences_per));
	}
	CUDA_FUNC_IN void setPassIndex(unsigned int idx)
	{
		m_passIdx = idx;
	}
	CUDA_FUNC_IN unsigned int getPassIndex() const
	{
		return m_passIdx;
	}
	CUDA_FUNC_IN unsigned int getNumSequences() const
	{
		return m_numSequences;
	}

	CUDA_FUNC_IN SamplerType operator()();
	CUDA_FUNC_IN SamplerType operator()(unsigned int idx);
	CUDA_FUNC_IN void operator()(const SamplerType& val)
	{
		unsigned int idx = getGlobalIdx_2D_2D();
		operator()(val, idx);
	}
	CUDA_FUNC_IN void operator()(const SamplerType& val, unsigned int idx)
	{
	}
};

struct SequenceSampler
{
private:
	SequenceSamplerData& data;
	unsigned int m_index;
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
		return f;
		//return fmodf(m_index / data.getNumRandomsPerSequence() + f, 1.0f) * 0.9999f;
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
		return f;
		//auto n = data.getNumRandomsPerSequenceSqrt();
		//int x = m_index % n, y = m_index / n;
		//float u = f.x + x / float(n), v = f.y + y / float(n);
		//return Vec2f(fmodf(u, 1.0f), fmodf(v, 1.0f)) * 0.9999f;
	}
public:
	CUDA_FUNC_IN SequenceSampler(unsigned int i, SequenceSamplerData& dat)
		: m_index(i), data(dat)
	{

	}
	CUDA_FUNC_IN float randomFloat()
	{
		float val = data.dim1(m_index++ % data.getNumSequences());
		return map(val) * 0.9999f;
	}
	CUDA_FUNC_IN Vec2f randomFloat2()
	{
		auto val = data.dim2(m_index++ % data.getNumSequences());
		return map(Vec2f(val.x, val.y)) * 0.9999f;
	}
	CUDA_FUNC_IN void StartSequence(unsigned int idx)
	{
		m_index = idx;
	}
};

SequenceSampler SequenceSamplerData::operator()()
{
	unsigned int idx = getGlobalIdx_2D_2D();
	return operator()(idx);
}
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
	void CopyToDevice()
	{

	}
	CUDA_FUNC_IN void setNumSequences(int val)
	{

	}
	CUDA_FUNC_IN RandomSampler& getSampler(unsigned int idx)
	{
#ifdef ISCUDA
		return m_pDeviceGenerators[idx % m_uNumGenerators];
#else
		return m_pHostGenerators[idx % m_uNumGenerators];
#endif
	}

	CUDA_FUNC_IN RandomSampler operator()() const
	{
		unsigned int idx = getGlobalIdx_2D_2D();
		return operator()(idx);
	}
	CUDA_FUNC_IN RandomSampler operator()(unsigned int idx) const
	{
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
		unsigned int idx = getGlobalIdx_2D_2D();
		operator()(val, idx);
	}
	CUDA_FUNC_IN void operator()(const RandomSampler& val, unsigned int idx)
	{
#ifdef ISCUDA
		if (idx < m_uNumGenerators)
			m_pDeviceGenerators[idx] = val;
#else
		m_pHostGenerators[idx % m_uNumGenerators] = val;
#endif
	}
};

typedef RandomSamplerData SamplerData;
//typedef SequenceSamplerData SamplerData;

typedef SamplerData::SamplerType Sampler;

class ISamplingSequenceGenerator
{
public:
	virtual ~ISamplingSequenceGenerator()
	{

	}
	virtual void Compute(float* sequence, unsigned int n) = 0;
	virtual void Compute(Vec2f* sequence, unsigned int n) = 0;
	void Compute(SequenceSamplerData& data)
	{
		Compute(&data.dim1(0), data.getNumSequences());
		Compute(&data.dim2(0), data.getNumSequences());
	}
	void Compute(RandomSamplerData& data)
	{

	}
};

}