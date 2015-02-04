#pragma once

#include "..\Kernel\k_Tracer.h"

class k_PrimTracer : public k_Tracer<false, false>
{
public:
	bool m_bDirect;
	k_PrimTracer()
		: m_bDirect(false)
	{
	}
	virtual void Debug(e_Image* I, const int2& pixel);
	virtual void CreateSliders(SliderCreateCallback a_Callback);
protected:
	virtual void DoRender(e_Image* I);
};