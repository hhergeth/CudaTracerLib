#pragma once

#include <Kernel/Tracer.h>
#include <CudaMemoryManager.h>
#include <Kernel/RayBuffer.h>

namespace CudaTracerLib {

struct rayData
{
	Spectrum throughput;
	short x, y;
	Spectrum L;
	Spectrum directF;
	float dDist;
	unsigned int dIdx;
};

typedef RayBuffer<rayData, 2> k_PTDBuffer;

class FastTracer : public Tracer<false, true>
{
public:
	PARAMETER_KEY(bool, PathTracingMode)
	FastTracer()
		: bufA(0), bufB(0), depthImage(0)
	{
		m_sParameters << KEY_PathTracingMode() << CreateSetBool(false);
	}
	~FastTracer()
	{
		Free();
	}
	void Free()
	{
		bufA->Free();
		bufB->Free();
		delete bufA;
		delete bufB;
	}
	virtual void Resize(unsigned int w, unsigned int h)
	{
		ThrowCudaErrors();
		Tracer<false, true>::Resize(w, h);
		ThrowCudaErrors();
		if (bufA)
			Free();
		bufA = new k_PTDBuffer(w * h);
		bufB = new k_PTDBuffer(w * h);
		ThrowCudaErrors();
	}
	void setDethImage(Image* img)
	{
		depthImage = img;
	}
protected:
	virtual void DoRender(Image* I);
private:
	k_PTDBuffer* bufA, *bufB;
	Image* depthImage;
	void doDirect(Image* I);
	void doPath(Image* I);
};

}