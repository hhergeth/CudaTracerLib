#pragma once

#include <Kernel/Tracer.h>
#include <Kernel/RayBuffer.h>
#include "VCMHelper.h"

namespace CudaTracerLib {

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

typedef RayBuffer<k_BPTPathState, 1> k_WVCM_LightBuffer;
typedef RayBuffer<k_BPTCamSubPathState, 1> k_WVCM_CamBuffer;

class WavefrontVCM : public Tracer<true, true>
{
public:
	WavefrontVCM(unsigned int a_NumLightRays = 1024 * 100);
protected:
	virtual void DoRender(Image* I);
	virtual void StartNewTrace(Image* I);
	virtual void RenderBlock(Image* I, int x, int y, int blockW, int blockH);
private:

	//current will be used for lookup, next will be stored in
	k_PhotonMapCollection<false, k_MISPhoton> m_sPhotonMapsNext;
	float m_fInitialRadius;
	unsigned long long m_uPhotonsEmitted;
	float getCurrentRadius(int exp)
	{
		return CudaTracerLib::getCurrentRadius(m_fInitialRadius, m_uPassesDone, exp);
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

}