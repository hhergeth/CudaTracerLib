#pragma once

#include <Kernel/Tracer.h>
#include <Kernel/DoubleRayBuffer.h>
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
	int last_light_vertex_start;
	unsigned int trace_res[MAX_LIGHT_SUB_PATH_LENGTH];
	Spectrum prev_throughput;
	BPTSubPathState state;
};

typedef DoubleRayBuffer<k_BPTPathState> k_WVCM_LightBuffer;
typedef DoubleRayBuffer<k_BPTCamSubPathState> k_WVCM_CamBuffer;

class WavefrontVCM : public Tracer<true>
{
public:
	CTL_EXPORT WavefrontVCM(unsigned int a_NumLightRays = 1024 * 100);
	CTL_EXPORT ~WavefrontVCM();
	CTL_EXPORT virtual float getSplatScale() const;
protected:
	CTL_EXPORT virtual void DoRender(Image* I);
	CTL_EXPORT virtual void StartNewTrace(Image* I);
	CTL_EXPORT virtual void RenderBlock(Image* I, int x, int y, int blockW, int blockH);
private:

	//current will be used for lookup, next will be stored in
	VCMSurfMap m_sPhotonMapsNext;
	float m_fInitialRadius;
	unsigned long long m_uPhotonsEmitted;
	float getCurrentRadius(int exp)
	{
		return CudaTracerLib::getCurrentRadius(m_fInitialRadius, m_uPassesDone, (float)exp);
	}
private:
	unsigned int m_uNumLightRays;
	k_WVCM_LightBuffer m_sLightBuf;
	BPTVertex* m_pDeviceLightVertices;
	k_WVCM_CamBuffer m_sCamBuf;
	unsigned int m_uLightOff;
	CTL_EXPORT void cppTest();
protected:
};

}