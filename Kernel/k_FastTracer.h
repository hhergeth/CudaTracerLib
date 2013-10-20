#pragma once

#include "k_Tracer.h"
#include "..\Base\CudaRandom.h"
#include "k_TraceHelper.h"

class k_RayIntersectKernel
{
public:
	unsigned int Length;
	traversalRay* m_pRayBuffer;
	traversalResult* m_pResultBuffer;
public:
	k_RayIntersectKernel(){}
	k_RayIntersectKernel(unsigned int n)
		: Length(n)
	{
		cudaMalloc(&m_pRayBuffer, sizeof(traversalRay) * Length);
		cudaMalloc(&m_pResultBuffer, sizeof(traversalResult) * Length);
	}
	CUDA_FUNC_IN traversalRay* getRayBuffer()
	{
		return m_pRayBuffer;
	}
	CUDA_FUNC_IN traversalResult* getResultBuffer()
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
	}
};

template<typename T, int N> class k_RayBuffer
{
private:
	T* m_pPayloadBuffer;
	k_RayIntersectKernel buffers[N];
	unsigned int Length;
public:
	k_RayBuffer(unsigned int n)
		: Length(n)
	{
		for(int i = 0; i < N; i++)
			buffers[i] = k_RayIntersectKernel(n);
		cudaMalloc(&m_pPayloadBuffer, sizeof(T) * Length);
	}
	CUDA_FUNC_IN k_RayIntersectKernel& operator[](int i)
	{
		return buffers[i];
	}
	CUDA_FUNC_IN T* getPayloadBuffer()
	{
		return m_pPayloadBuffer;
	}
	void Free()
	{
		for(int i = 0; i < N; i++)
			buffers[i].Free();
		cudaFree(m_pPayloadBuffer);
	}
	void ClearRays()
	{
		cudaMemset(m_pPayloadBuffer, 0, sizeof(T) * Length);
		for(int i = 0; i < N; i++)
			buffers[i].ClearRays();
	}
	void ClearResults()
	{
		for(int i = 0; i < N; i++)
			buffers[i].ClearResults();
	}
	void IntersectBuffers(unsigned int n, int i = -1, bool skipOuterTree = false)
	{
		if(i == -1)
			for(int i = 0; i < N; i++)
				buffers[i].IntersectBuffers(n, skipOuterTree);
		else buffers[i].IntersectBuffers(n, skipOuterTree);
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
	virtual void Resize(unsigned int w, unsigned int h)
	{
		k_ProgressiveTracer::Resize(w, h);
		if(hostRays)
			cudaFreeHost(hostRays);
		if(hostResults)
			cudaFreeHost(hostResults);
		cudaMallocHost(&hostRays, sizeof(traversalRay) * w * h);
		cudaMallocHost(&hostResults, sizeof(traversalResult) * w * h);
		if(intersector)
			intersector->Free();
		intersector = new k_RayBuffer<rayData, 2>(w * h);
	}
protected:
	virtual void DoRender(e_Image* I);
private:
	k_RayBuffer<rayData, 2>* intersector;
	void doDirect(e_Image* I);
	void doPath(e_Image* I);
};