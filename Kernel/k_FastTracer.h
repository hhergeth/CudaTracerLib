#pragma once

#include "k_Tracer.h"
#include "..\Base\CudaRandom.h"
#include "k_TraceHelper.h"

template<typename T> class k_RayIntersectKernel
{
public:
	unsigned int Length;
	traversalRay* m_pRayBuffer;
	T* m_pPayloadBuffer;
	traversalResult* m_pResultBuffer;
public:
	k_RayIntersectKernel(unsigned int n)
		: Length(n)
	{
		cudaMalloc(&m_pRayBuffer, sizeof(traversalRay) * Length);
		cudaMalloc(&m_pResultBuffer, sizeof(traversalResult) * Length);
		cudaMalloc(&m_pPayloadBuffer, sizeof(T) * Length);
	}
	traversalRay* getRayBuffer()
	{
		return m_pRayBuffer;
	}
	T* getPayloadBuffer()
	{
		return m_pPayloadBuffer;
	}
	traversalResult* getResultBuffer()
	{
		return m_pResultBuffer;
	}
	void IntersectBuffers(unsigned int N, bool skipOuterTree = false)
	{
		if(N > Length)
		{
			//I'd worry cause you ve written to invalid memory
			throw 1;
		}
		__internal__IntersectBuffers(N, m_pRayBuffer, m_pResultBuffer, skipOuterTree);
	}
	void Free()
	{
		cudaFree(m_pPayloadBuffer);
		cudaFree(m_pRayBuffer);
		cudaFree(m_pResultBuffer);
	}
	void ClearResults()
	{
		cudaMemset(m_pResultBuffer, 0, sizeof(traversalResult) * Length);
	}
	void ClearRays()
	{
		cudaMemset(m_pRayBuffer, 0, sizeof(traversalRay) * Length);
		cudaMemset(m_pPayloadBuffer, 0, sizeof(T) * Length);
	}
};

class k_FastTracer : public k_ProgressiveTracer
{
public:
	struct rayData
	{
		Spectrum L;
		Spectrum throughput;
		short x,y;
	};
	traversalRay* hostRays;
	traversalResult* hostResults;
	k_FastTracer()
		: intersector(0), hostRays(0), hostResults(0)
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