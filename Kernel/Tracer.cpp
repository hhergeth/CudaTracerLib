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
	if (m_pBlockSampler)
		delete m_pBlockSampler;
	if (m_pPixelVarianceBuffer)
	{
		m_pPixelVarianceBuffer->Free();
		delete m_pPixelVarianceBuffer;
	}
	m_pBlockSampler = 0;
	m_pPixelVarianceBuffer = 0;
}

void TracerBase::setCorrectSamplingSequenceGenerator()
{
	auto new_type = m_sParameters.getValue(KEY_SamplingSequenceType());
	ISamplingSequenceGenerator* new_gen = 0;

	if (new_type == SamplingSequenceGeneratorTypes::Independent && dynamic_cast<IndependantSamplingSequenceGenerator*>(m_pSamplingSequenceGenerator) == 0)
		new_gen = new IndependantSamplingSequenceGenerator();

	if (new_gen)
	{
		delete m_pSamplingSequenceGenerator;
		m_pSamplingSequenceGenerator = new_gen;
	}
}

}