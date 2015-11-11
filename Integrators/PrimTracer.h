#pragma once

#include <Kernel/Tracer.h>

namespace CudaTracerLib {

class PrimTracer : public Tracer<false, false>
{
	Spectrum* m_pDeviceLastImage1, *m_pDeviceLastImage2;
	Sensor lastSensor;
public:
	bool m_bDirect;
	Image* depthImage;
	PrimTracer();
	virtual void Debug(Image* I, const Vec2i& pixel);
	virtual void CreateSliders(SliderCreateCallback a_Callback) const;
	virtual void Resize(unsigned int _w, unsigned int _h);
protected:
	virtual void DoRender(Image* I);
};

}