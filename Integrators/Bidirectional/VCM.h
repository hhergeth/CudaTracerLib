#pragma once

#include <Kernel/Tracer.h>
#include "VCMHelper.h"

namespace CudaTracerLib {

class VCM : public Tracer<true, true>
{
public:
	VCM();
protected:
	virtual void DoRender(Image* I);
	virtual void StartNewTrace(Image* I);
	virtual void RenderBlock(Image* I, int x, int y, int blockW, int blockH);
private:
	//current will be used for lookup, next will be stored in
	k_PhotonMapCollection<false, k_MISPhoton> m_sPhotonMapsCurrent, m_sPhotonMapsNext;
	float m_fInitialRadius;
	unsigned long long m_uPhotonsEmitted;
	float getCurrentRadius(int exp)
	{
		return CudaTracerLib::getCurrentRadius(m_fInitialRadius, m_uPassesDone, exp);
	}
};

}