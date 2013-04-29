#pragma once

#include "k_Tracer.h"
#include "..\Base\CudaRandom.h"

struct BufferEntry
{
	float3 Pos;
	float3 Dir;
	TraceResult Res;
	short x, y;
	float3 cl;
	float3 cf;
	CUDA_FUNC_IN BufferEntry(float3 p, float3 d, int _x, int _y)
	{
		Pos = p;
		Dir = d;
		Res.Init();
		x = (short)_x;
		y = (short)_y;
		cl = make_float3(0);
		cf = make_float3(1);
	}
};

class k_MultiPathTracer : public k_RandTracerBase
{
private:
	float4* m_pTmpData;
	BufferEntry* m_pBuffer;
	unsigned int m_uLastCount;
public:
	k_MultiPathTracer()
		: k_RandTracerBase()
	{
		m_pTmpData = 0;
		m_pBuffer = 0;
	}
	virtual ~k_MultiPathTracer()
	{
		cudaFree(m_pTmpData);
		cudaFree(m_pBuffer);
	}
	virtual void Resize(unsigned int _w, unsigned int _h);
	virtual void Debug(int2 pixel);
protected:
	virtual void DoRender(RGBCOL* a_Buf);
	virtual void StartNewTrace();
};