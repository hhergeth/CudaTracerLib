#pragma once

#include "k_Tracer.h"
#include "..\Base\CudaRandom.h"

class k_PathTracer : public k_TracerBase
{
public:
	k_PathTracer()
		: k_TracerBase()
	{
		
	}
	virtual ~k_PathTracer()
	{
		
	}
	virtual void Resize(unsigned int _w, unsigned int _h);
	virtual void Debug(int2 pixel);
protected:
	virtual void DoRender(e_Image* I);
	virtual void StartNewTrace(e_Image* I);
};