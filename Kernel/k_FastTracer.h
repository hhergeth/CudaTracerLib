#pragma once

#include "k_Tracer.h"
#include "..\Base\CudaRandom.h"
#include "k_TraceHelper.h"

template<typename T> class k_RayIntersectKernel
{
private:
	unsigned int offset, Length;
	T* m_pRayBuffer;
	TraceResult* m_pResultBuffer;
public:
	k_RayIntersectKernel(unsigned int n, unsigned int o)
		: offset(o), Length(n)
	{
		cudaMalloc(&m_pRayBuffer, sizeof(T) * Length);
		cudaMalloc(&m_pResultBuffer, sizeof(TraceResult) * Length);
	}
	T* getRayBuffer()
	{
		return m_pRayBuffer;
	}
	TraceResult* getResultBuffer()
	{
		return m_pResultBuffer;
	}
	void IntersectBuffers(unsigned int N)
	{
		if(N > Length)
		{
			//I'd worry cause you ve written to invalid memory
			throw 1;
		}
		__internal__IntersectBuffers(N, m_pRayBuffer, m_pResultBuffer, sizeof(T), offset);
	}
	void Free()
	{
		cudaFree(m_pRayBuffer);
		cudaFree(m_pResultBuffer);
	}
	void ClearResults()
	{
		cudaMemset(m_pResultBuffer, 0, sizeof(TraceResult) * Length);
	}
	void ClearRays()
	{
		cudaMemset(m_pRayBuffer, 0, sizeof(T) * Length);
	}
};

class k_FastTracer : public k_ProgressiveTracer
{
public:
	struct rayData
	{
		Ray r;
		Spectrum L;
		Spectrum throughput;
		short x,y;
	};
	k_FastTracer()
		: intersector(0)
	{
	}
	virtual void Resize(unsigned int w, unsigned int h);
protected:
	virtual void DoRender(e_Image* I);
private:
	k_RayIntersectKernel<rayData>* intersector;
	void doDirect(e_Image* I);
	void doPath(e_Image* I);
};