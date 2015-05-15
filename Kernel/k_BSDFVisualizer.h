#pragma once

#include "..\Kernel\k_Tracer.h"
#include "..\Base\CudaRandom.h"
#include "..\Engine\e_BSDF.h"
#include "..\Engine\e_Light.h"

class k_BSDFVisualizer : public k_Tracer<false, true>
{
public:
	Vec3f m_wo;
	BSDFALL* m_Bsdf;
	e_InfiniteLight* m_pLight;
	e_Stream<char>* m_pBuffer;
	e_Buffer<e_MIPMap, e_KernelMIPMap>* m_pBuffer2;
	e_MIPMap* m_pMipMap;
	float LScale;
	bool cosTheta;
	bool drawEnvMap;
	k_BSDFVisualizer()
		: m_wo(Vec3f(0, 0, 1)), m_Bsdf(0), LScale(1), cosTheta(true), m_pBuffer(0), m_pBuffer2(0), drawEnvMap(false), m_pLight(0), m_pMipMap(0)
	{
	}
	~k_BSDFVisualizer()
	{
		if (m_pBuffer)
			m_pBuffer->Free();
		if (m_pBuffer2)
			m_pBuffer2->Free();
		if (m_pLight)
			delete m_pLight;
	}
	virtual void Debug(e_Image* I, const Vec2i& pixel, ITracerDebugger* debugger = 0);
	void DrawRegion(e_Image* I, const Vec2i& off, const Vec2i& size);
	void setSkydome(const char* compiledPath);
protected:
	virtual void DoRender(e_Image* I);
	virtual void StartNewTrace(e_Image* I)
	{
		I->Clear();
	}
};