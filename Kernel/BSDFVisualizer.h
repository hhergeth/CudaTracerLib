#pragma once

#include <Kernel/Tracer.h>
#include <Engine/Light.h>

namespace CudaTracerLib {

struct KernelMIPMap;
class MIPMap;
template<typename H, typename D> class Buffer;
template<typename T> class Stream;

class BSDFVisualizer : public Tracer<false, true>
{
public:
	NormalizedT<Vec3f> m_wo;
	BSDFALL* m_Bsdf;
	InfiniteLight* m_pLight;
	Stream<char>* m_pBuffer;
	Buffer<MIPMap, KernelMIPMap>* m_pBuffer2;
	MIPMap* m_pMipMap;
	float LScale;
	bool cosTheta;
	bool drawEnvMap;
	BSDFVisualizer()
		: m_wo(0.0f, 0.0f, 1.0f), m_Bsdf(0), LScale(1), cosTheta(true), m_pBuffer(0), m_pBuffer2(0), drawEnvMap(false), m_pLight(0), m_pMipMap(0)
	{
	}
	CTL_EXPORT ~BSDFVisualizer();
	CTL_EXPORT virtual void Debug(Image* I, const Vec2i& pixel);
	CTL_EXPORT void DrawRegion(Image* I, const Vec2i& off, const Vec2i& size);
	CTL_EXPORT void setSkydome(const char* compiledPath);
protected:
	CTL_EXPORT virtual void DoRender(Image* I);
	virtual void StartNewTrace(Image* I)
	{
		I->Clear();
	}
};

}