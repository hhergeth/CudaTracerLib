#pragma once

#include <Kernel/Tracer.h>
#include <Kernel/PathSpaceFilteringBuffer.h>

namespace CudaTracerLib {

class GameTracer : public Tracer<false>, public IDepthTracer
{
	PathSpaceFilteringBuffer buf;
public:
	GameTracer()
		: buf(256 * 256)
	{
		m_sParameters.addChildParameterCollection("PathSpaceFilterBuffer", &buf.getParameterCollection());

		buf.getParameterCollection().setValue(PathSpaceFilteringBuffer::KEY_UseRadius_PixelFootprintSize(), false);
		buf.getParameterCollection().setValue(PathSpaceFilteringBuffer::KEY_PrevFrameAlphaIndirect(), 0.05f);
	}
	CTL_EXPORT virtual void Resize(unsigned int _w, unsigned int _h);
protected:
	CTL_EXPORT virtual void DoRender(Image* I);
	CTL_EXPORT virtual void DebugInternal(Image* I, const Vec2i& pixel);
};

}