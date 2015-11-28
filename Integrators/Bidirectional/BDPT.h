#pragma once

#include <Kernel/Tracer.h>

namespace CudaTracerLib {

class BDPT : public Tracer<true, true>
{
public:
	PARAMETER_KEY(bool, UseMis)
	PARAMETER_KEY(int, Force_s)
	PARAMETER_KEY(int, Force_t)
	PARAMETER_KEY(float, ResultMultiplier)

	BDPT()
	{
		m_sParameters << KEY_UseMis() << CreateSetBool(true)
					  << KEY_Force_s() << CreateInterval<int>(-1, -1, INT_MAX)
					  << KEY_Force_t() << CreateInterval<int>(-1, -1, INT_MAX)
					  << KEY_ResultMultiplier() << CreateParameter(1.0f);
	}
	virtual void Debug(Image* I, const Vec2i& pixel);
protected:
	virtual void RenderBlock(Image* I, int x, int y, int blockW, int blockH);
};

}