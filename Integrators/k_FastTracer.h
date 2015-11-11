#pragma once

#include <Kernel/k_Tracer.h>
#include <CudaMemoryManager.h>
#include <Kernel/k_RayBuffer.h>

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

typedef k_RayBuffer<rayData, 2> k_PTDBuffer;

class k_FastTracer : public k_Tracer<false, true>
{
public:
	bool pathTracer;
	k_FastTracer(bool doPT = false)
		: bufA(0), bufB(0), pathTracer(doPT), depthImage(0)
	{

	}
	~k_FastTracer()
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
		k_Tracer<false, true>::Resize(w, h);
		ThrowCudaErrors();
		if (bufA)
			Free();
		bufA = new k_PTDBuffer(w * h);
		bufB = new k_PTDBuffer(w * h);
		ThrowCudaErrors();
	}
	void setDethImage(e_Image* img)
	{
		depthImage = img;
	}
protected:
	virtual void DoRender(e_Image* I);
private:
	k_PTDBuffer* bufA, *bufB;
	e_Image* depthImage;
	void doDirect(e_Image* I);
	void doPath(e_Image* I);
};

}