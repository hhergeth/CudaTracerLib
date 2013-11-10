#pragma once

#include "k_Tracer.h"
#include "..\Base\CudaRandom.h"

class k_BDPT : public k_ProgressiveTracer
{
public:
	k_BDPT(bool direct = false)
	{
	}
	virtual void Debug(int2 pixel);
protected:
	virtual void DoRender(e_Image* I);
};