#pragma once

#include <Kernel/Tracer.h>

namespace CudaTracerLib {

class GameTracer : public Tracer<false>, public IDepthTracer
{
	Spectrum* m_pDeviceLastDirectImage1, *m_pDeviceLastDirectImage2;
	Spectrum* m_pDeviceLastIndirectImage1, *m_pDeviceLastIndirectImage2;
	Sensor lastSensor;
	int iterations;
public:
	GameTracer()
		: m_pDeviceLastDirectImage1(0), m_pDeviceLastDirectImage2(0),
		  m_pDeviceLastIndirectImage1(0), m_pDeviceLastIndirectImage2(0), iterations(0)
	{
		
	}
	CTL_EXPORT virtual void Resize(unsigned int _w, unsigned int _h);
protected:
	CTL_EXPORT virtual void DoRender(Image* I);
	CTL_EXPORT virtual void DebugInternal(Image* I, const Vec2i& pixel);
};

}