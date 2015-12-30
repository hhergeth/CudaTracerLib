#pragma once

#include <Kernel/Tracer.h>
#include <CudaMemoryManager.h>
#include <Kernel/RayBuffer.h>
#include <Math/half.h>

namespace CudaTracerLib {

struct WavefrontPTRayData
{
	Spectrum throughput;
	half x, y;
	Spectrum L;
	Spectrum directF;
	float dDist;
	unsigned int dIdx;
};

typedef RayBuffer<WavefrontPTRayData, 2> WavefrontPathTracerBuffer;

class WavefrontPathTracer : public Tracer<false, true>, public IDepthTracer
{
public:
	PARAMETER_KEY(bool, Direct)
	PARAMETER_KEY(int, MaxPathLength)

	WavefrontPathTracer()
		: bufA(0), bufB(0)
	{
		m_sParameters << KEY_Direct() << CreateSetBool(true) << KEY_MaxPathLength() << CreateInterval<int>(7, 1, INT_MAX);
	}
	~WavefrontPathTracer()
	{
		bufA->Free(); delete bufA;
		bufB->Free(); delete bufB;
	}
	virtual void Resize(unsigned int w, unsigned int h)
	{
		Tracer<false, true>::Resize(w, h);
		if (bufA)
		{
			bufA->Free(); delete bufA;
			bufB->Free(); delete bufB;
		}
		bufA = new WavefrontPathTracerBuffer(w * h);
		bufB = new WavefrontPathTracerBuffer(w * h);
	}
protected:
	virtual void DoRender(Image* I);
private:
	WavefrontPathTracerBuffer* bufA, *bufB;
};

}