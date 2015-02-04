#pragma once

#include "..\Kernel\k_Tracer.h"
#include "..\Base\CudaRandom.h"
#include "..\Kernel\k_TraceHelper.h"
#include "..\Engine\e_Core.h"

template<typename T, int N> struct k_RayBuffer
{
private:
	traversalRay* m_pRayBuffer[N];
	traversalResult* m_pResultBuffer[N];
	T* m_pPayloadBuffer;
	unsigned int m_cInsertCounters[N];
	unsigned int Length;
public:
	CUDA_FUNC_IN k_RayBuffer(){};

	k_RayBuffer(unsigned int Length)
		: Length(Length)
	{
		for(int i = 0; i < N; i++)
		{
			CUDA_MALLOC(&m_pRayBuffer[i], sizeof(traversalRay) * Length);
			CUDA_MALLOC(&m_pResultBuffer[i], sizeof(traversalResult) * Length);
		}
		CUDA_MALLOC(&m_pPayloadBuffer, sizeof(T) * Length);
	}
	void Free()
	{
		for(int i = 0; i < N; i++)
		{
			CUDA_FREE(m_pRayBuffer[i]);
			CUDA_FREE(m_pResultBuffer[i]);
		}
		CUDA_FREE(m_pPayloadBuffer);
	}
	template<bool ANY_HIT> unsigned int IntersectBuffers(bool skipOuterTree = false)
	{
		unsigned int S = 0;
		for (int i = 0; i < N; i++)
		{
			S += m_cInsertCounters[i];
			__internal__IntersectBuffers(m_cInsertCounters[i], m_pRayBuffer[i], m_pResultBuffer[i], skipOuterTree, ANY_HIT);
		}
		return S;
	}
	void Clear()
	{
		for (int i = 0; i < N; i++)
		{
			cudaMemset(m_pRayBuffer[i], 0, sizeof(traversalRay) * Length);
			cudaMemset(m_pResultBuffer[i], 0, sizeof(traversalResult) * Length);
		}
		cudaMemset(m_pPayloadBuffer, 0, sizeof(T) * Length);
		for (int i = 0; i < N; i++)
			m_cInsertCounters[i] = 0;
	}
	void setNumRays(unsigned int num, unsigned int bufIdx = -1)
	{
		if (bufIdx == -1)
			for (int i = 0; i < N; i++)
				m_cInsertCounters[i] = num;
		else m_cInsertCounters[bufIdx] = num;
	}
	unsigned int getNumRays(unsigned int bufIdx = -1)
	{
		if (bufIdx == -1)
		{
			unsigned int s = 0;
			for (int i = 0; i < N; i++)
				s += m_cInsertCounters[i];
			return s;
		}
		else return m_cInsertCounters[bufIdx];
	}
	CUDA_FUNC_IN unsigned int insertRay(unsigned int bufIdx)
	{
		unsigned int i = Platform::Increment(m_cInsertCounters + bufIdx);
		return i;
	}
	CUDA_FUNC_IN traversalRay& operator()(unsigned int i, unsigned int j)
	{
		return m_pRayBuffer[j][i];
	}
	CUDA_FUNC_IN traversalResult& res(unsigned int i, unsigned int j)
	{
		return m_pResultBuffer[j][i];
	}
	CUDA_FUNC_IN T& operator()(unsigned int i)
	{
		return m_pPayloadBuffer[i];
	}
};

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
	k_FastTracer()
		: bufA(0), bufB(0)
	{
		
	}
	virtual void Resize(unsigned int w, unsigned int h)
	{
		k_Tracer<false, true>::Resize(w, h);
		ThrowCudaErrors();
		if(bufA)
		{
			bufA->Free();
			bufB->Free();
			delete bufA;
			delete bufB;
		}
		bufA = new k_PTDBuffer(w * h);
		bufB = new k_PTDBuffer(w * h);
		ThrowCudaErrors();
	}
protected:
	virtual void DoRender(e_Image* I);
private:
	k_PTDBuffer* bufA, *bufB;
	void doDirect(e_Image* I);
	void doPath(e_Image* I);
};