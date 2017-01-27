#include <StdAfx.h>
#include "Tracer.h"
#include "TraceHelper.h"
#include "Sampler.h"
#include "BlockSampler/UniformBlockSampler.h"
#include "BlockSampler/VarianceBlockSampler.h"

namespace CudaTracerLib {

TracerBase::TracerBase()
	: m_pScene(0), m_pBlockSampler(new VarianceBlockSampler(0, 0)), m_pSamplingSequenceGenerator(0)
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
}

void UpdateSamplingSequenceGenerator(SamplingSequenceGeneratorTypes type, ISamplingSequenceGenerator*& gen)
{

}

}