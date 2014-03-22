#pragma once

#include "..\Base\CudaRandom.h"
#include "..\MathTypes.h"
#include "..\Engine\e_Image.h"

class k_BlockSampler
{
public:
	struct pixelData
	{
		//float I, I2;
		float V;
	};
private:
	int m_uTargetHeight;
	int m_uBlockSize;
	int m_uNumBlocks;
	int m_uBlockDimX,m_uBlockDimY;
	float* m_pDeviceBlockData;
	unsigned int* m_pDeviceIndexData;
	unsigned int* m_pHostIndexData;
	unsigned int m_uNumBlocksToLaunch;
	bool hasValidData;

	pixelData* cudaPixels;
	unsigned int nSamplesDone;
public:
	k_BlockSampler()
		: m_uBlockSize(32), hasValidData(false)
	{
		m_uNumBlocks = -1;
		m_pDeviceBlockData = 0;
		m_pDeviceIndexData = 0;
		cudaPixels = 0;
		nSamplesDone = 0;
		m_pHostIndexData = 0;
	}
	void AddPass(const e_Image& img);
	void Initialize(unsigned int w, unsigned int h);
	unsigned int NumBlocks()
	{
		return m_uNumBlocksToLaunch;
	}
	CUDA_FUNC_IN unsigned int blockIdx(unsigned int unmappedIdx)
	{
#ifdef ISCUDA
		return hasValidData ? m_pDeviceIndexData[unmappedIdx] : unmappedIdx;
#else
		return hasValidData ? m_pHostIndexData[unmappedIdx] : unmappedIdx;
#endif
	}
	CUDA_FUNC_IN uint2 blockCoord(unsigned int unmappedIdx)
	{
		unsigned int idx = blockIdx(unmappedIdx);
		unsigned int x = idx % m_uBlockDimX, y = idx / m_uBlockDimX;
		return make_uint2(x, y);
	}
	CUDA_FUNC_IN uint2 blockCoord()
	{
#ifdef ISCUDA
		unsigned int unmappedIdx = ::blockIdx.x;
#else
		unsigned int unmappedIdx = 0;
#endif
		return blockCoord(unmappedIdx);
	}
	CUDA_FUNC_IN uint2 pixelCoord()
	{
#ifdef ISCUDA
		uint2 tc = make_uint2(threadIdx.x, threadIdx.y);
#else
		uint2 tc = make_uint2(0, 0);
#endif
		uint2 bc = blockCoord();
		return make_uint2(bc.x * 32 + tc.x, bc.y * 32 + tc.y);
	}
	CUDA_HOST dim3 blockDim()
	{
		return dim3(m_uNumBlocksToLaunch);
	}
	CUDA_HOST dim3 threadDim()
	{
		return dim3(32, 32, 1);
	}
	CUDA_FUNC_IN void getRect(unsigned int idx, unsigned int& x, unsigned int& y, unsigned int& w, unsigned int& h)
	{
		w = h = m_uBlockSize;
		uint2 q = blockCoord(idx);
		x = q.x * m_uBlockSize;
		y = m_uTargetHeight - q.y * m_uBlockSize - 1 - h;
	}
};