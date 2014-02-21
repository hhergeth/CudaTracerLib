#pragma once

#include "k_Tracer.h"
#include "..\Base\CudaRandom.h"
#include "k_TraceHelper.h"
#include "..\Engine\e_Core.h"

template<typename T, int N> struct k_RayBuffer
{
private:
	traversalRay* m_pRayBuffer[N];
	traversalResult* m_pResultBuffer[N];
	T* m_pPayloadBuffer;
	unsigned int* m_cInsertCounter;
	unsigned int Length;
public:
	k_RayBuffer(unsigned int Length)
		: Length(Length)
	{
		for(int i = 0; i < N; i++)
		{
			CUDA_MALLOC(&m_pRayBuffer[i], sizeof(traversalRay) * Length);
			CUDA_MALLOC(&m_pResultBuffer[i], sizeof(traversalResult) * Length);
		}
		CUDA_MALLOC(&m_pPayloadBuffer, sizeof(T) * Length);
		CUDA_MALLOC(&m_cInsertCounter, sizeof(unsigned int));
	}
	void Free()
	{
		for(int i = 0; i < N; i++)
		{
			CUDA_FREE(m_pRayBuffer[i]);
			CUDA_FREE(m_pResultBuffer[i]);
		}
		CUDA_FREE(m_pPayloadBuffer);
		CUDA_FREE(m_cInsertCounter);
	}
	template<bool ANY_HIT> unsigned int IntersectBuffers(bool doAll = true, bool skipOuterTree = false)
	{
		unsigned int n = getCreatedRayCount();
		if(n > Length)
		{
			//I'd worry cause you ve written to invalid memory
			throw 1;
		}
		int a = doAll ? N : 1;
		for(int i = 0; i < a; i++)
			__internal__IntersectBuffers(n, m_pRayBuffer[i], m_pResultBuffer[i], skipOuterTree, ANY_HIT);
		return a * n;
	}
	void StartNewTraversal()
	{
		setGeneratedRayCount(0);
	}
	unsigned int getCreatedRayCount()
	{
		unsigned int r;
		cudaMemcpy(&r, m_cInsertCounter, 4, cudaMemcpyDeviceToHost);
		return r;
	}
	void setGeneratedRayCount(unsigned int N)
	{
		cudaMemcpy(m_cInsertCounter, &N, 4, cudaMemcpyHostToDevice);
	}
	CUDA_FUNC_IN CUDA_HOST unsigned int insertRay()
	{
		unsigned int i = Platform::Increment(m_cInsertCounter);
		return i;
	}
	CUDA_FUNC_IN CUDA_HOST traversalRay& operator()(unsigned int i, unsigned int j)
	{
		return m_pRayBuffer[j][i];
	}
	CUDA_FUNC_IN CUDA_HOST traversalResult& res(unsigned int i, unsigned int j)
	{
		return m_pResultBuffer[j][i];
	}
	CUDA_FUNC_IN CUDA_HOST T& operator()(unsigned int i)
	{
		return m_pPayloadBuffer[i];
	}
};

template<typename T, int N, int M> struct k_RayBufferManager
{
	k_RayBuffer<T, N>* buffers[M];
	unsigned int idx;

	k_RayBufferManager(unsigned int Length)
		: idx(0)
	{
		for(int i = 0; i < M; i++)
			buffers[i] = new k_RayBuffer<T, N>(Length);
	}

	void Free()
	{
		for(int i = 0; i < M; i++)
		{
			buffers[i]->Free();
			delete buffers[i];
		}
	}

	k_RayBuffer<T, N>* current()
	{
		return buffers[idx];
	}

	k_RayBuffer<T, N>* next()
	{
		idx = (idx + 1) % M;
		return buffers[idx];
	}
};

/*
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
		CUDA_MALLOC(&m_pRayBuffer, sizeof(traversalRay) * Length);
		CUDA_MALLOC(&m_pResultBuffer, sizeof(traversalResult) * Length);
		CUDA_MALLOC(&m_pIndexBuffer, sizeof(unsigned int) * Length);
		CUDA_MALLOC(&m_cInsertCounter, sizeof(unsigned int));
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
		CUDA_FREE(m_pRayBuffer);
		CUDA_FREE(m_pResultBuffer);
		CUDA_FREE(m_pIndexBuffer);
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
		unsigned int i = Platform::Increment(m_cInsertCounter);
		if(a_RayIndex)
			*a_RayIndex = i;
		m_pIndexBuffer[i] = payloadIdx;
		if(a_Out)
			*a_Out = m_pResultBuffer + i;
		return m_pRayBuffer + i;
	}
	CUDA_FUNC_IN CUDA_HOST traversalRay* InsertRay(unsigned int payloadIdx, unsigned int rayIdx, traversalResult** a_Out = 0)
	{
		unsigned int i = Platform::Increment(m_cInsertCounter);
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
		CUDA_MALLOC(&m_pPayloadBuffer, sizeof(T) * Length);
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
		CUDA_FREE(m_pPayloadBuffer);
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
*/
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
	//traversalRay* hostRays;
	//traversalResult* hostResults;
	k_FastTracer()
		: intersector(0)//, hostRays(0), hostResults(0)
	{
		
	}
	virtual void Resize(unsigned int w, unsigned int h)
	{
		k_ProgressiveTracer::Resize(w, h);
		//if(hostRays)
		//	CUDA_FREEHost(hostRays);
		//if(hostResults)
		//	CUDA_FREEHost(hostResults);
		//CUDA_MALLOCHost(&hostRays, sizeof(traversalRay) * w * h);
		//CUDA_MALLOCHost(&hostResults, sizeof(traversalResult) * w * h);
		//if(intersector)
		//	intersector->Free();
		ThrowCudaErrors();
		if(intersector)
		{
			intersector->Free();
			delete intersector;
		}
		intersector = new k_RayBufferManager<rayData, 2, 2>(w * h);
		ThrowCudaErrors();
	}
	virtual void Debug(int2 pixel);
protected:
	virtual void DoRender(e_Image* I);
private:
	k_RayBufferManager<rayData, 2, 2>* intersector;
	void doDirect(e_Image* I);
	void doPath(e_Image* I);
};