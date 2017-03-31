#pragma once

#include <Kernel/Tracer.h>
#include <Base/CudaMemoryManager.h>
#include <Kernel/RayBuffer.h>

namespace CudaTracerLib {

struct EmptyRayData
{

};

typedef RayBuffer<EmptyRayData, 2> FastTracerBuffer;

class FastTracer : public Tracer<false>, public IDepthTracer
{
public:
	FastTracer()
		: bufA(0)
	{
	}
	~FastTracer()
	{
		bufA->Free();
	}
	virtual void Resize(unsigned int w, unsigned int h)
	{
		Tracer<false>::Resize(w, h);
		if (bufA)
			delete bufA;
		bufA = new FastTracerBuffer(w * h);
	}
protected:
	CTL_EXPORT virtual void DoRender(Image* I);
private:
	FastTracerBuffer* bufA;
};

}