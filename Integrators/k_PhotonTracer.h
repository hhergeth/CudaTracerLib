#pragma once

#include "..\Kernel\k_Tracer.h"
#include "..\Base\CudaRandom.h"
#include "..\Base\StringUtils.h"

class k_PhotonTracer : public k_ProgressiveTracer
{
public:
	k_PhotonTracer()
		:N (256 * 256 * 4)
	{
	}
	virtual void Debug(int2 pixel);
	void PrintStatus(std::vector<std::string>& a_Buf)
	{
		double pC = floor((double)(m_uPassesDone * N) / 1000000.0);
		a_Buf.push_back(format("Photons emitted : %d[Mil]", (int)pC));
	}
protected:
	virtual void DoRender(e_Image* I);
	unsigned int N;
};