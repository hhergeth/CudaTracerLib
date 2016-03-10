#pragma once

#include <Math/Vector.h>
#include <Base/CudaRandom.h>
#include <CudaMemoryManager.h>

//#define SEQUENCE_SAMPLER

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

struct SequenceSampler;
template<int N_SEQUENCES, int SEQUENCE_LENGTH> struct SequenceSamplerData
{

	enum { NUM_SEQUENCES = N_SEQUENCES, LEN_SEQUENCE = SEQUENCE_LENGTH };

	template<typename T> using Sequence = SequenceSamplerData_Sequence<T, SEQUENCE_LENGTH>;

private:
	Sequence<float> m_d1Data[N_SEQUENCES];
	Sequence<float2> m_d2Data[N_SEQUENCES];
	float m_mapping_parameter;
public:
	CUDA_FUNC_IN SequenceSamplerData() = default;
	SequenceSamplerData(float mapping_parameter)
		: m_mapping_parameter(mapping_parameter)
	{

	}
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
	CUDA_FUNC_IN float getMappingParameter() const
	{
		return m_mapping_parameter;
	}

	CUDA_FUNC_IN SequenceSampler operator()() const;
	CUDA_FUNC_IN void operator()(const SequenceSampler& val)
	{

	}
};

#ifdef SEQUENCE_SAMPLER

typedef SequenceSamplerData<5, 25> SamplerData;
typedef SequenceSampler Sampler;

#else

struct RandomSamplerData
{
private:
	unsigned int m_uNumGenerators;
	CudaRNG* m_pHostGenerators;
	CudaRNG* m_pDeviceGenerators;
public:
	CUDA_FUNC_IN RandomSamplerData() = default;
	RandomSamplerData(unsigned int n)
		: m_uNumGenerators(n)
	{
		CUDA_MALLOC(&m_pDeviceGenerators, m_uNumGenerators * sizeof(CudaRNG));
		m_pHostGenerators = new CudaRNG[m_uNumGenerators];
		Generate();
	}
	void Generate()
	{
		for (unsigned int i = 0; i < m_uNumGenerators; i++)
		{
			m_pHostGenerators[i] = CudaRNG(i);
		}
		CUDA_MEMCPY_TO_DEVICE(m_pDeviceGenerators, m_pHostGenerators, sizeof(CudaRNG) * m_uNumGenerators);
	}
	void Free()
	{
		CUDA_FREE(m_pDeviceGenerators);
		m_pDeviceGenerators = 0;
		delete[] m_pHostGenerators;
		m_pHostGenerators = 0;
	}
	CUDA_FUNC_IN CudaRNG operator()() const
	{
		unsigned int idx = getGlobalIdx_2D_2D();
		unsigned int i = idx % m_uNumGenerators;
#ifdef ISCUDA
		CudaRNG rng = m_pDeviceGenerators[i];
		if (idx >= m_uNumGenerators)
		{
			//skipahead_sequence(idx - m_uNumGenerators, &rng.state);
		}
#else
		CudaRNG rng = m_pHostGenerators[i];
#endif
		return rng;
	}
	CUDA_FUNC_IN void operator()(const CudaRNG& val)
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
typedef CudaRNG Sampler;

#endif


SamplerData ConstructDefaultSamplerData();
extern CUDA_ALIGN(16) CUDA_CONST SamplerData g_SamplerDataDevice;
CTL_EXPORT extern CUDA_ALIGN(16) SamplerData g_SamplerDataHost;
#ifdef ISCUDA
#define g_SamplerData g_SamplerDataDevice
#else
#define g_SamplerData g_SamplerDataHost
#endif

#ifdef SEQUENCE_SAMPLER
struct SequenceSampler
{
private:
	unsigned int m_index;
	unsigned short d1, d2;
	CUDA_FUNC_IN float map(float f) const
	{
		float i = (float)(m_index / SamplerData::NUM_SEQUENCES);
		return fmod(f + i * g_SamplerData.getMappingParameter(), 1.0f);
	}
public:
	CUDA_FUNC_IN SequenceSampler(unsigned int i)
		: m_index(i), d1(0), d2(0)
	{

	}
	CUDA_FUNC_IN float randomFloat()
	{
		float val = g_SamplerData.dim1(m_index % SamplerData::NUM_SEQUENCES)[d1++];
		return map(val);
	}
	CUDA_FUNC_IN Vec2f randomFloat2()
	{
		auto val = g_SamplerData.dim2(m_index % SamplerData::NUM_SEQUENCES)[d2++];
		return Vec2f(map(val.x), map(val.y));
	}
};

template<int N_SEQUENCES, int SEQUENCE_LENGTH> SequenceSampler SequenceSamplerData<N_SEQUENCES, SEQUENCE_LENGTH>::operator()() const
{
	return SequenceSampler(getGlobalIdx_2D_2D());
}

#endif

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
};

}