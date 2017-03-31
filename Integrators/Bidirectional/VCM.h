#pragma once

#include <Kernel/Tracer.h>
#include "VCMHelper.h"

namespace CudaTracerLib {

class VCM : public Tracer<true>
{
public:
	CTL_EXPORT VCM();
protected:
	CTL_EXPORT virtual void DoRender(Image* I);
	CTL_EXPORT virtual void StartNewTrace(Image* I);
	CTL_EXPORT virtual void RenderBlock(Image* I, int x, int y, int blockW, int blockH);
private:
	//current will be used for lookup, next will be stored in
	VCMSurfMap m_sPhotonMapsCurrent, m_sPhotonMapsNext;
	float m_fInitialRadius;
	unsigned long long m_uPhotonsEmitted;
	float getCurrentRadius(int exp)
	{
		return CudaTracerLib::getCurrentRadius(m_fInitialRadius, m_uPassesDone, (float)exp);
	}
};

}