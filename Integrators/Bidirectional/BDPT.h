#pragma once

#include <Kernel/Tracer.h>

namespace CudaTracerLib {

class k_BDPT : public Tracer<true, true>
{
public:
	k_BDPT()
		: force_s(-1), force_t(-1), use_mis(true), LScale(1)
	{
	}
	virtual void Debug(Image* I, const Vec2i& pixel);
	bool use_mis;
	int force_s, force_t;
	float LScale;
protected:
	virtual void RenderBlock(Image* I, int x, int y, int blockW, int blockH);
};

}