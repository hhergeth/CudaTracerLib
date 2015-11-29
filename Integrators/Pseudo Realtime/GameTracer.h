#pragma once

#include <Kernel/Tracer.h>

namespace CudaTracerLib {

class GameTracer : public Tracer<false, false>, public IDepthTracer
{
	Spectrum* m_pDeviceLastImage1, *m_pDeviceLastImage2;
	Sensor lastSensor;
	int iterations;
public:
	GameTracer()
		: m_pDeviceLastImage1(0), m_pDeviceLastImage2(0), iterations(0)
	{
		
	}
	virtual void Debug(Image* I, const Vec2i& pixel);
	virtual void Resize(unsigned int _w, unsigned int _h);
protected:
	virtual void DoRender(Image* I);
};

}