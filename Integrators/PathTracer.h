#pragma once

#include <Kernel/Tracer.h>

namespace CudaTracerLib {

class PathTracer : public Tracer<true, true>
{
public:
	bool m_Direct, m_Regularization;
	PathTracer(bool direct = false, bool regularization = false)
		: m_Direct(direct), m_Regularization(regularization)
	{
	}
	virtual void Debug(Image* I, const Vec2i& pixel);
protected:
	virtual void RenderBlock(Image* I, int x, int y, int blockW, int blockH);
};

}