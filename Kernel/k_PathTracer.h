#pragma once

#include "k_Tracer.h"
#include "..\Base\CudaRandom.h"

class k_PathTracer : public k_RandTracerBase
{
private:
	float4* m_pTmpData;
public:
	k_PathTracer()
		: k_RandTracerBase()
	{
		m_pTmpData = 0;
	}
	virtual ~k_PathTracer()
	{
		cudaFree(m_pTmpData);
	}
	virtual void Resize(unsigned int _w, unsigned int _h);
	virtual void Debug(int2 pixel);
protected:
	virtual void DoRender(RGBCOL* a_Buf);
	virtual void StartNewTrace(RGBCOL* a_Buf);
};