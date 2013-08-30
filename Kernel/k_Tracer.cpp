#include <StdAfx.h>
#include "k_Tracer.h"
#include "k_TraceHelper.h"

k_TracerRNGBuffer k_Tracer::g_sRngs;
static bool initrng = false;

void k_Tracer::InitRngs(unsigned int N)
{
	if(!initrng)
	{
		initrng = 1;
		g_sRngs = k_TracerRNGBuffer(N);
	}
}

float k_TracerBase::getValuePerSecond(float val, float invScale)
{
	return val / (m_fTimeSpentRendering * invScale);
}

float k_TracerBase::getTimeSpentRendering()
{
	return m_fTimeSpentRendering;
}

void k_TracerRNGBuffer::createGenerators(unsigned int a_Spacing, unsigned int a_Offset)
{
	for(int i = 0; i < m_uNumGenerators; i++)
		((CudaRNG*)m_pHostGenerators + i)->Initialize(i, a_Spacing, a_Offset);
	cudaMemcpy(m_pDeviceGenerators, m_pHostGenerators, sizeof(curandState) * m_uNumGenerators, cudaMemcpyHostToDevice);
}