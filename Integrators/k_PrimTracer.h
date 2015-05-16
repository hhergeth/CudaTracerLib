#pragma once

#include "..\Kernel\k_Tracer.h"

class k_PrimTracer : public k_Tracer<false, false>
{
public:
	bool m_bDirect;
	e_Image* depthImage;
	k_PrimTracer();
	virtual void Debug(e_Image* I, const Vec2i& pixel);
	virtual void CreateSliders(SliderCreateCallback a_Callback) const;
protected:
	virtual void DoRender(e_Image* I);
};