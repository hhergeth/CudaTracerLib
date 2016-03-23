#include <StdAfx.h>
#include "Tracer.h"
#include "TraceHelper.h"
#include "BlockSampler.h"
#include "Sampler.h"

namespace CudaTracerLib {

TracerBase::TracerBase()
	: m_pScene(0), m_pBlockSampler(0), m_pSamplingSequenceGenerator(0)
{
	ThrowCudaErrors(cudaEventCreate(&start));
	ThrowCudaErrors(cudaEventCreate(&stop));
	m_sParameters << KEY_SamplerActive() << CreateSetBool(false)
				  << KEY_SamplingSequenceType() << SamplingSequenceGeneratorTypes::Independent;
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

BlockSampleImage TracerBase::getDeviceBlockSampler() const
{
	return getBlockSampler()->getBlockImage();
}

void TracerBase::allocateBlockSampler(Image* I)
{
	m_pBlockSampler = new BlockSampler(I);
}

}