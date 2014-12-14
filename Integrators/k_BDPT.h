#pragma once

#include "..\Kernel\k_Tracer.h"
#include "..\Base\CudaRandom.h"

class k_BDPT : public k_ProgressiveTracer
{
public:
	k_BDPT(bool direct = false)
		: force_s(-1), force_t(-1), use_mis(true), LScale(1)
	{
	}
	virtual void Debug(e_Image* I, int2 pixel);
	bool use_mis;
	int force_s, force_t;
	float LScale;
protected:
	virtual void DoRender(e_Image* I);
};