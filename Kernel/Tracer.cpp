#include <StdAfx.h>
#include "Tracer.h"
#include "TraceHelper.h"
#include "Sampler.h"
#include "BlockSampler/UniformBlockSampler.h"
#include "BlockSampler/VarianceBlockSampler.h"
#include "Sampler.h"

namespace CudaTracerLib {

TracerBase::TracerBase()
	: m_pScene(0), m_pBlockSampler(new VarianceBlockSampler(0, 0)), m_pSamplingSequenceGenerator(0), m_pPixelVarianceBuffer(0)
{
	ThrowCudaErrors(cudaEventCreate(&start));
	ThrowCudaErrors(cudaEventCreate(&stop));
	m_sParameters << KEY_SamplingSequenceType() << SamplingSequenceGeneratorTypes::Independent;
	setCorrectSamplingSequenceGenerator();
}

TracerBase::~TracerBase()
{
	if (start == 0)
	{
		std::cout << "Calling ~TracerBase() multiple times!\n";
		return;
	}
	ThrowCudaErrors(cudaEventDestroy(start));
	ThrowCudaErrors(cudaEventDestroy(stop));
	start = stop = 0;
	if (m_pSamplingSequenceGenerator)
	{
		delete m_pSamplingSequenceGenerator;
		m_pSamplingSequenceGenerator = 0;
	}
	if (m_pBlockSampler)
		delete m_pBlockSampler;
	if (m_pPixelVarianceBuffer)
	{
		m_pPixelVarianceBuffer->Free();
		delete m_pPixelVarianceBuffer;
	}
	m_pBlockSampler = 0;
	m_pPixelVarianceBuffer = 0;
	m_debugVisualizerManager.Free();
}

void TracerBase::generateNewRandomSequences()
{
	GenerateNewRandomSequences(*m_pSamplingSequenceGenerator);
}

template<typename T> struct check_type
{
	void operator()(ISamplingSequenceGenerator*& gen, SamplingSequenceGeneratorTypes new_type, SamplingSequenceGeneratorTypes T_type)
	{
		if (dynamic_cast<SamplingSequenceGeneratorHost<T>*>(gen) == 0 && new_type == T_type)
		{
			delete gen;
			gen = new SamplingSequenceGeneratorHost<T>();
		}
	}
};
void TracerBase::setCorrectSamplingSequenceGenerator()
{
	auto new_type = m_sParameters.getValue(KEY_SamplingSequenceType());

	check_type<IndependantSamplingSequenceGenerator>()(m_pSamplingSequenceGenerator, new_type, Independent);
	check_type<StratifiedSamplingSequenceGenerator>()(m_pSamplingSequenceGenerator, new_type, Stratified);
}

}