#pragma once

#include "k_Tracer.h"

class k_PrimTracer : public k_RandTracerBase
{
public:
	k_PrimTracer()
	{
	}
	virtual void Debug(int2 pixel);
protected:
	virtual void DoRender(RGBCOL* a_Buf);
};