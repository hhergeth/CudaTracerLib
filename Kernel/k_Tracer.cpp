#include <StdAfx.h>
#include "k_Tracer.h"
#include "k_TraceHelper.h"
#include "k_BlockSampler.h"

CudaRNGBuffer k_TracerBase::g_sRngs;
static bool initrng = false;

void k_TracerBase::InitRngs(unsigned int N)
{
	if(!initrng)
	{
		initrng = 1;
		g_sRngs = CudaRNGBuffer(N);
	}
}

k_TracerBase::k_TracerBase()
	: m_pScene(0), m_pBlockSampler(0)
{
	InitRngs(1024 * 768);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
}

k_BlockSampleImage k_TracerBase::getDeviceBlockSampler() const
{
	return getBlockSampler()->getBlockImage();
}

void k_TracerBase::allocateBlockSampler(e_Image* I)
{
	m_pBlockSampler = new k_BlockSampler(I);
}
