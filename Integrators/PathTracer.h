#pragma once

#include <Kernel/Tracer.h>

namespace CudaTracerLib {

class PathTracer : public Tracer<true>
{
public:
	PARAMETER_KEY(bool, Direct)
	PARAMETER_KEY(bool, Regularization)
	PARAMETER_KEY(int, MaxPathLength)
	PARAMETER_KEY(int, RRStartDepth)
	PathTracer()
	{
		m_sParameters << KEY_Direct()				<< CreateSetBool(true)
					  << KEY_Regularization()		<< CreateSetBool(false)
					  << KEY_MaxPathLength()		<< CreateInterval<int>(7, 1, INT_MAX)
					  << KEY_RRStartDepth()			<< CreateInterval(5, 1, INT_MAX);
	}
protected:
	CTL_EXPORT virtual void RenderBlock(Image* I, int x, int y, int blockW, int blockH);
	CTL_EXPORT virtual void DebugInternal(Image* I, const Vec2i& pixel);
};

}