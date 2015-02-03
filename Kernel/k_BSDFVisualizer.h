#pragma once

#include "..\Kernel\k_Tracer.h"
#include "..\Base\CudaRandom.h"
#include "..\Engine\e_BSDF.h"
#include "..\Engine\e_Light.h"

class k_BSDFVisualizer : public k_ProgressiveTracer
{
public:
	float3 m_wo;
	BSDFALL* m_Bsdf;
	e_InfiniteLight* m_pLight;
	e_Stream<char>* m_pBuffer;
	e_Buffer<e_MIPMap, e_KernelMIPMap>* m_pBuffer2;
	e_MIPMap* m_pMipMap;
	float LScale;
	bool cosTheta;
	bool drawEnvMap;
	k_BSDFVisualizer()
		: m_wo(make_float3(0, 0, 1)), m_Bsdf(0), LScale(1), cosTheta(true), m_pBuffer(0), m_pBuffer2(0), drawEnvMap(false)
	{
		k_Tracer::InitRngs();
	}
	void Debug(const int2& pixel);
	void DrawRegion(e_Image* I, int2 off, int2 size);
	void setSkydome(const char* compiledPath);
protected:
	virtual void DoRender(e_Image* I);
};