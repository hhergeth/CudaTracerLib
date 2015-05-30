#pragma once

#include "..\Kernel\k_Tracer.h"
#include "../Engine/e_Light.h"

struct e_KernelMIPMap;
class e_MIPMap;
template<typename H, typename D> class e_Buffer;
template<typename T> class e_Stream;

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
	~k_BSDFVisualizer();
	virtual void Debug(e_Image* I, const Vec2i& pixel);
	void DrawRegion(e_Image* I, const Vec2i& off, const Vec2i& size);
	void setSkydome(const char* compiledPath);
protected:
	virtual void DoRender(e_Image* I);
	virtual void StartNewTrace(e_Image* I)
	{
		I->Clear();
	}
};