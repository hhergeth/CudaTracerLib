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

typedef k_RayBuffer<k_BPTPathState, 1> k_WVCM_LightBuffer;
typedef k_RayBuffer<k_BPTCamSubPathState, 1> k_WVCM_CamBuffer;

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
		return ::getCurrentRadius(m_fInitialRadius, m_uPassesDone, exp);
	}
private:
	unsigned int m_uNumLightRays;
	k_WVCM_LightBuffer m_sLightBufA, m_sLightBufB;
	BPTVertex* m_pDeviceLightVertices;
	k_WVCM_CamBuffer m_sCamBufA, m_sCamBufB;
	unsigned int m_uLightOff;
	void cppTest();
protected:
	virtual float getSplatScale();
};