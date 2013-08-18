#include <StdAfx.h>
#include "k_Tracer.h"

//no idea why but fewer rngs will introduce fireflies....
k_RandTracerBase::k_RandTracerBase()
		: k_TracerBase(), m_sRngs(1 << 13)
{

}

void k_TracerBase::StartNewTrace(e_Image* I)
{
	//I->StartNewRendering();
}

float k_TracerBase::getValuePerSecond(float val, float invScale)
{
	return val / (m_fTimeSpentRendering * invScale);
}