#pragma once

#include <Kernel/Tracer.h>

namespace CudaTracerLib {

class PhotonTracer : public Tracer<true>
{
public:
	PARAMETER_KEY(bool, CorrectDifferentials)
	PhotonTracer()
	{
		m_sParameters << KEY_CorrectDifferentials() << CreateSetBool(false);
	}
	virtual void PrintStatus(std::vector<std::string>& a_Buf) const
	{
		float nPhotons = math::floor((float)(m_uPassesDone * w * h) / 1000000.0f);
		a_Buf.push_back(format("Photons emitted : %d[Mil]", (int)nPhotons));
		a_Buf.push_back(format("Photons per second : %f[Mil]", nPhotons / m_fAccRuntime));
	}
protected:
	CTL_EXPORT virtual void DoRender(Image* I);
	CTL_EXPORT virtual void DebugInternal(Image* I, const Vec2i& pixel);
};

}