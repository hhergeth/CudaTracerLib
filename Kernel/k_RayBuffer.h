#pragma once

#include "../Defines.h"
#include "../CudaMemoryManager.h"

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
		for (int i = 0; i < N; i++)
		{
			CUDA_MALLOC(&m_pRayBuffer[i], sizeof(traversalRay) * Length);
			CUDA_MALLOC(&m_pResultBuffer[i], sizeof(traversalResult) * Length);
		}
		CUDA_MALLOC(&m_pPayloadBuffer, sizeof(T) * Length);
		for (int i = 0; i < N; i++)
			m_cInsertCounters[i] = 0;
	}
	void Free()
	{
		for (int i = 0; i < N; i++)
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
			if (m_cInsertCounters[i])
				__internal__IntersectBuffers(m_cInsertCounters[i], m_pRayBuffer[i], m_pResultBuffer[i], skipOuterTree, ANY_HIT);
		}
		return S;
	}
	void Clear()
	{
		for (int i = 0; i < N; i++)
		{
			ThrowCudaErrors(cudaMemset(m_pRayBuffer[i], 0, sizeof(traversalRay) * Length));
			ThrowCudaErrors(cudaMemset(m_pResultBuffer[i], 0, sizeof(traversalResult) * Length));
		}
		ThrowCudaErrors(cudaMemset(m_pPayloadBuffer, 0, sizeof(T) * Length));
		for (int i = 0; i < N; i++)
			m_cInsertCounters[i] = 0;
	}
	void setNumRays(unsigned int num, unsigned int bufIdx = 0xffffffff)
	{
		if (bufIdx == 0xffffffff)
			for (int i = 0; i < N; i++)
				m_cInsertCounters[i] = num;
		else m_cInsertCounters[bufIdx] = num;
	}
	unsigned int getNumRays(unsigned int bufIdx = 0xffffffff)
	{
		if (bufIdx == 0xffffffff)
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
	//inserts for all active lanes a ray into the buffer
	CUDA_DEVICE unsigned int insertRayCUDA_LANES(unsigned int bufIdx)
	{
		unsigned int* pCounter = m_cInsertCounters + bufIdx;

		const bool          terminated = 1;//nodeAddr == EntrypointSentinel;
		const unsigned int  maskTerminated = __ballot(terminated);
		const int           numTerminated = __popc(maskTerminated);
		const int           idxTerminated = __popc(maskTerminated & ((1u << threadIdx.x) - 1));

		__shared__ volatile unsigned int insertBase;
		if (idxTerminated == 0)
			insertBase = atomicAdd(pCounter, (unsigned int)numTerminated);
		return insertBase + idxTerminated;
	}
};