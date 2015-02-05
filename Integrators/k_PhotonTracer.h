#pragma once

#include "..\Kernel\k_Tracer.h"
#include "..\Base\CudaRandom.h"
#include "..\Base\StringUtils.h"

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
		a_Buf.push_back(format("Photons emitted : %d[Mil]", (int)pC));
	}
protected:
	virtual void DoRender(e_Image* I);
};