#include <StdAfx.h>
#include "k_Tracer.h"
#include "k_TraceHelper.h"

CudaRNGBuffer k_Tracer::g_sRngs;
static bool initrng = false;

void k_Tracer::InitRngs(unsigned int N)
{
	if(!initrng)
	{
		initrng = 1;
		g_sRngs = CudaRNGBuffer(N);
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

void k_ProgressiveTracer::StartNewTrace(e_Image* I)
{
	I->Clear();
	m_uPassesDone = 0; 
}

void k_ProgressiveTracer::DoRender(e_Image* I)
{
	m_uNumRaysTraced = 0;
	//m_fTimeSpentRendering = 0;
}