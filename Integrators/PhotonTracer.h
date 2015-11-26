#pragma once

#include <Kernel/Tracer.h>

namespace CudaTracerLib {

class PhotonTracer : public Tracer<false, true>
{
	bool m_bCorrectDifferentials;
public:
	PhotonTracer()
		: m_bCorrectDifferentials(false)
	{
	}
	virtual void Debug(Image* I, const Vec2i& pixel);
	virtual void PrintStatus(std::vector<std::string>& a_Buf) const
	{
		double nPhotons = math::floor((double)(m_uPassesDone * w * h) / 1000000.0);
		a_Buf.push_back(format("Photons emitted : %d[Mil]", (int)nPhotons));
		a_Buf.push_back(format("Photons per second : %f[Mil]", nPhotons / m_fAccRuntime));
	}
protected:
	virtual void DoRender(Image* I);
};

}