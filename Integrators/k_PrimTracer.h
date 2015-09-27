#pragma once

#include "..\Kernel\k_Tracer.h"

class k_PrimTracer : public k_Tracer<false, false>
{
	Spectrum* m_pDeviceLastImage1, *m_pDeviceLastImage2;
	e_Sensor lastSensor;
public:
	bool m_bDirect;
	e_Image* depthImage;
	k_PrimTracer();
	virtual void Debug(e_Image* I, const Vec2i& pixel);
	virtual void CreateSliders(SliderCreateCallback a_Callback) const;
	virtual void Resize(unsigned int _w, unsigned int _h);
protected:
	virtual void DoRender(e_Image* I);
};