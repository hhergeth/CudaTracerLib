#pragma once

#include "k_Tracer.h"

class k_PrimTracer : public k_RandTracerBase
{
public:
	k_PrimTracer()
	{
	}
	virtual void Debug(int2 pixel);
	virtual void CreateSliders(SliderCreateCallback a_Callback);
protected:
	virtual void DoRender(e_Image* I);
};