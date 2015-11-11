#pragma once

#include <Kernel/k_Tracer.h>
#include "k_VCMHelper.h"

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
	k_PhotonMapCollection<false, k_MISPhoton> m_sPhotonMapsCurrent, m_sPhotonMapsNext;
	float m_fInitialRadius;
	unsigned long long m_uPhotonsEmitted;
	float getCurrentRadius(int exp)
	{
		return ::getCurrentRadius(m_fInitialRadius, m_uPassesDone, exp);;
	}
};