#pragma once

#include "..\Kernel\k_Tracer.h"
#include "..\Base\CudaRandom.h"
#include "k_PhotonMapHelper.h"

class k_VCM : public k_Tracer<true, true>
{
public:
	k_VCM();
protected:
	virtual void DoRender(e_Image* I);
	virtual void StartNewTrace(e_Image* I);
	virtual void RenderBlock(e_Image* I, int x, int y, int blockW, int blockH);
private:
	//current will be used for lookup, next will be stored in
	k_PhotonMapCollection<false> m_sPhotonMapsCurrent, m_sPhotonMapsNext;
	float m_fInitialRadius;
	unsigned long long m_uPhotonsEmitted;
	float getCurrentRadius(int exp)
	{
		float f = 1;
		for (unsigned int k = 1; k < m_uPassesDone; k++)
			f *= (k + ALPHA) / k;
		f = math::pow(m_fInitialRadius, float(exp)) * f * 1.0f / float(m_uPassesDone);
		return math::pow(f, 1.0f / float(exp));
	}
};