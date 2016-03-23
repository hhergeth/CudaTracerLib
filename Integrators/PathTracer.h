#pragma once

#include <Kernel/Tracer.h>

namespace CudaTracerLib {

class PathTracer : public Tracer<true, true>
{
public:
	PARAMETER_KEY(bool, Direct)
	PARAMETER_KEY(bool, Regularization)
	PathTracer()
	{
		m_sParameters << KEY_Direct() << CreateSetBool(true) << KEY_Regularization() << CreateSetBool(false);
	}
protected:
	CTL_EXPORT virtual void RenderBlock(Image* I, int x, int y, int blockW, int blockH);
	CTL_EXPORT virtual void DebugInternal(Image* I, const Vec2i& pixel);
};

}