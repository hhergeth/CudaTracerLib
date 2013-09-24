#pragma once

#include "k_Tracer.h"
#include "..\Base\CudaRandom.h"

class k_PhotonTracer : public k_ProgressiveTracer
{
public:
	k_PhotonTracer()
	{
	}
protected:
	virtual void DoRender(e_Image* I);
};