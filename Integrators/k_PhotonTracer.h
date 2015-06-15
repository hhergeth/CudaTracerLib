#pragma once

#include "..\Kernel\k_Tracer.h"

class k_PhotonTracer : public k_Tracer<false, true>
{
public:
	k_PhotonTracer()
	{
	}
	virtual void Debug(e_Image* I, const Vec2i& pixel);
	virtual void PrintStatus(std::vector<std::string>& a_Buf) const
	{
		double pC = floor((double)(m_uPassesDone * w * h) / 1000000.0);
		int q = (int)pC;
		a_Buf.push_back(format("Photons emitted : %d[Mil]", q));
	}
protected:
	virtual void DoRender(e_Image* I);
};