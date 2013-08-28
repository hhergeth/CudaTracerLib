#pragma once

#include "k_Tracer.h"
#include "..\Base\CudaRandom.h"

class k_PathTracer : public k_ProgressiveTracer
{
public:
	bool m_Direct;
	k_PathTracer(bool direct = false)
		: m_Direct(direct)
	{
	}
	virtual void Debug(int2 pixel);
protected:
	virtual void DoRender(e_Image* I);
};