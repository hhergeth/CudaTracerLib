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
	unsigned int* m_pIndexBuffer;
	unsigned int* m_cInsertCounter;
public:
	k_RayIntersectKernel(){}
	k_RayIntersectKernel(unsigned int n)
		: Length(n)
	{
		cudaMalloc(&m_pRayBuffer, sizeof(traversalRay) * Length);
		cudaMalloc(&m_pResultBuffer, sizeof(traversalResult) * Length);
		cudaMalloc(&m_pIndexBuffer, sizeof(unsigned int) * Length);
		cudaMalloc(&m_cInsertCounter, sizeof(unsigned int));
	}
	template<bool ANY_HIT> unsigned int IntersectBuffers(bool skipOuterTree = false)
	{
		unsigned int N = getCreatedRayCount();
		if(N > Length)
		{
			//I'd worry cause you ve written to invalid memory
			throw 1;
		}
		__internal__IntersectBuffers(N, m_pRayBuffer, m_pResultBuffer, skipOuterTree, ANY_HIT);
		return N;
	}
	void Free()
	{
		cudaFree(m_pRayBuffer);
		cudaFree(m_pResultBuffer);
		cudaFree(m_pIndexBuffer);
	}
	void StartNewTraversal()
	{
		unsigned int zero = 0;
		cudaMemcpy(m_cInsertCounter, &zero, 4, cudaMemcpyHostToDevice);
	}
	unsigned int getCreatedRayCount()
	{
		unsigned int r;
		cudaMemcpy(&r, m_cInsertCounter, 4, cudaMemcpyDeviceToHost);
		return r;
	}
	CUDA_FUNC_IN CUDA_HOST traversalRay* InsertRay(unsigned int payloadIdx, unsigned int* a_RayIndex = 0, traversalResult** a_Out = 0)
	{
#ifdef ISCUDA
		unsigned int i = atomicInc(m_cInsertCounter, 0xffffffff);
#else
		unsigned int i = InterlockedIncrement(m_cInsertCounter);
#endif
		if(a_RayIndex)
			*a_RayIndex = i;
		m_pIndexBuffer[i] = payloadIdx;
		if(a_Out)
			*a_Out = m_pResultBuffer + i;
		return m_pRayBuffer + i;
	}
	CUDA_FUNC_IN CUDA_HOST traversalRay* InsertRay(unsigned int payloadIdx, unsigned int rayIdx, traversalResult** a_Out = 0)
	{
#ifdef ISCUDA
		unsigned int i = atomicInc(m_cInsertCounter, 0xffffffff);
#else
		unsigned int i = InterlockedIncrement(m_cInsertCounter);
#endif
		m_pIndexBuffer[rayIdx] = payloadIdx;
		if(a_Out)
			*a_Out = m_pResultBuffer + rayIdx;
		return m_pRayBuffer + rayIdx;
	}
	CUDA_FUNC_IN unsigned int GetPayloadIndex(unsigned int workIdx)
	{
		return m_pIndexBuffer[workIdx];
	}
	CUDA_FUNC_IN CUDA_HOST traversalRay* FetchRay(unsigned int i, traversalResult** a_Out = 0)
	{
		if(a_Out)
			*a_Out = m_pResultBuffer + i;
		return m_pRayBuffer + i;
	} 
};

template<typename T, int N> class k_RayBuffer
{
private:
	T* m_pPayloadBuffer;
	k_RayIntersectKernel buffers[N];
	unsigned int Length;
public:
	k_RayBuffer(){}
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
	CUDA_FUNC_IN T& operator()(unsigned int i)
	{
		return m_pPayloadBuffer[i];
	}
	void Free()
	{
		for(int i = 0; i < N; i++)
			buffers[i].Free();
		cudaFree(m_pPayloadBuffer);
	}
	void StartNewRendering()
	{
		cudaMemset(m_pPayloadBuffer, 0, sizeof(T) * Length);
		StartNewTraversal();
	}
	void StartNewTraversal()
	{
		for(int i = 0; i < N; i++)
			buffers[i].StartNewTraversal();
	}
	template<bool ANY_HIT> unsigned int IntersectBuffers( int i = -1, bool skipOuterTree = false)
	{
		int r = 0;
		if(i == -1)
			for(int i = 0; i < N; i++)
				r += buffers[i].IntersectBuffers<ANY_HIT>(skipOuterTree);
		else r += buffers[i].IntersectBuffers<ANY_HIT>(skipOuterTree);
		return r;
	}
	unsigned int getCreatedRayCount()
	{
		unsigned int r = 0;
		for(int i = 0; i < N; i++)
			r += buffers[i].getCreatedRayCount();
		return r;
	}
};

class k_FastTracer : public k_ProgressiveTracer
{
public:
	struct rayData
	{
		Spectrum D;
		float dDist;
		Spectrum L;
		Spectrum throughput;
		unsigned int dIndex;
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