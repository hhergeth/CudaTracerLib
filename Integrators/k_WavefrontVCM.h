#pragma once

#include "..\Kernel\k_Tracer.h"
#include "../Kernel/k_RayBuffer.h"
#include "k_VCMHelper.h"

//in vertices
#define MAX_LIGHT_SUB_PATH_LENGTH 6

struct k_BPTPathState
{
	unsigned int m_uVertexStart;
	BPTSubPathState state;
};

struct k_BPTCamSubPathState
{
	int x, y;
	Spectrum acc;
	BPTSubPathState state;
};

class k_WavefrontVCM : public k_Tracer<true, true>
{
public:
	k_WavefrontVCM(unsigned int a_NumLightRays = 1024 * 100);
protected:
	virtual void DoRender(e_Image* I);
	virtual void StartNewTrace(e_Image* I);
	virtual void RenderBlock(e_Image* I, int x, int y, int blockW, int blockH);
private:

	//current will be used for lookup, next will be stored in
	k_PhotonMapCollection<false, k_MISPhoton> m_sPhotonMapsNext;
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
private:
	unsigned int m_uNumLightRays;
	k_RayBuffer<k_BPTPathState, 1> m_sLightBufA, m_sLightBufB;
	BPTVertex* m_pDeviceLightVertices;
	k_RayBuffer<k_BPTCamSubPathState, 1> m_sCamBufA, m_sCamBufB;
	unsigned int m_uLightOff;
	void cppTest();
protected:
	virtual float getSplatScale();
};