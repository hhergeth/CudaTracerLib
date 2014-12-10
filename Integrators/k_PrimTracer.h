#pragma once

#include "..\Kernel\k_Tracer.h"

class k_PrimTracer : public k_OnePassTracer
{
public:
	bool m_bDirect;
	k_PrimTracer()
		: m_bDirect(false)
	{
	}
	virtual void Debug(e_Image* I, int2 pixel);
	virtual void CreateSliders(SliderCreateCallback a_Callback);
protected:
	virtual void DoRender(e_Image* I);
};