#include <StdAfx.h>
#include "Tracer.h"
#include "TraceHelper.h"
#include "BlockSampler.h"

namespace CudaTracerLib {

CudaRNGBuffer TracerBase::g_sRngs;
static bool initrng = false;

void TracerBase::InitRngs(unsigned int N)
{
	if (!initrng)
	{
		initrng = 1;
		g_sRngs = CudaRNGBuffer(N);
	}
}

TracerBase::TracerBase()
	: m_pScene(0), m_pBlockSampler(0)
{
	InitRngs(1024 * 768);
	ThrowCudaErrors(cudaEventCreate(&start));
	ThrowCudaErrors(cudaEventCreate(&stop));
	m_sParameters << KEY_SamplerActive() << CreateSetBool(false);
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