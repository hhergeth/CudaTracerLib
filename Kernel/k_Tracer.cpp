#include <StdAfx.h>
#include "k_Tracer.h"

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

void k_TracerBase::StartNewTrace(e_Image* I)
{
	//I->StartNewRendering();
}

float k_TracerBase::getValuePerSecond(float val, float invScale)
{
	return val / (m_fTimeSpentRendering * invScale);
}