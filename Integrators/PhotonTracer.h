#pragma once

#include <Kernel/Tracer.h>

namespace CudaTracerLib {

class k_PhotonTracer : public Tracer<false, true>
{
public:
	k_PhotonTracer()
	{
	}
	virtual void Debug(Image* I, const Vec2i& pixel);
	virtual void PrintStatus(std::vector<std::string>& a_Buf) const
	{
		double pC = math::floor((double)(m_uPassesDone * w * h) / 1000000.0);
		int q = (int)pC;
		a_Buf.push_back(format("Photons emitted : %d[Mil]", q));
	}
protected:
	virtual void DoRender(Image* I);
};

}