#pragma once

#include "k_Tracer.h"

class k_PrimTracer : public k_TracerBase
{
public:
	bool m_bDirect;
	k_PrimTracer()
		: m_bDirect(false)
	{
	}
	virtual void Debug(int2 pixel);
	virtual void CreateSliders(SliderCreateCallback a_Callback);
protected:
	virtual void DoRender(e_Image* I);
};