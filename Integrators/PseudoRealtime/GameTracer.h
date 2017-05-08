#pragma once

#include <Kernel/Tracer.h>
#include <Kernel/PathSpaceFilteringBuffer.h>

namespace CudaTracerLib {

class GameTracer : public Tracer<false>, public IDepthTracer
{
	PathSpaceFilteringBuffer buf;
public:
	GameTracer()
		: buf(1)
	{
		m_sParameters.addChildParameterCollection("PathSpaceFilterBuffer", &buf.getParameterCollection());
	}
	CTL_EXPORT virtual void Resize(unsigned int _w, unsigned int _h);
protected:
	CTL_EXPORT virtual void DoRender(Image* I);
	CTL_EXPORT virtual void DebugInternal(Image* I, const Vec2i& pixel);
};

}