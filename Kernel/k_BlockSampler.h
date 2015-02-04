#pragma once

#include "..\Base\CudaRandom.h"
#include "..\MathTypes.h"
#include "..\Engine\e_Image.h"

#define threadsPerBlock dim3(16, 8)
#define numBlocks dim3(4, 8)
#define blockSize 64

struct k_SamplerPixelData
{
	float E, E2;
	float n;
	float min, max;
};

struct k_BlockSampleImage
{
	e_Image img;
	k_SamplerPixelData* m_pLumData;
	unsigned int w;

	k_BlockSampleImage(e_Image* img, k_SamplerPixelData* lumData)
		: img(*img), m_pLumData(lumData)
	{
		unsigned int y;
		img->getExtent(w, y);
	}

	CUDA_DEVICE CUDA_HOST void Add(int x, int y, const Spectrum& c);
};

class k_BlockSampler
{
	k_SamplerPixelData* m_pLumData;
	e_Image* img;

	unsigned int m_uPassesDone;
	unsigned int* m_pDeviceIndexData;
	float* m_pDeviceBlockData;
	float* m_pHostBlockData;
	unsigned int* m_pHostIndexData;
	unsigned int m_uNumBlocksToLaunch;
	bool hasValidData;
	unsigned int* m_pDeviceSamplesData;
	unsigned int* m_pHostSamplesData;
public:
	k_BlockSampler(e_Image* img)
		: img(img)
	{
		CUDA_MALLOC(&m_pLumData, img->getWidth() * img->getHeight() * sizeof(k_SamplerPixelData));
		CUDA_MALLOC(&m_pDeviceBlockData, totalNumBlocks() * sizeof(float));
		CUDA_MALLOC(&m_pDeviceIndexData, totalNumBlocks() * sizeof(unsigned int));
		CUDA_MALLOC(&m_pDeviceSamplesData, totalNumBlocks() * sizeof(unsigned int));
		m_pHostIndexData = new unsigned int[totalNumBlocks()];
		m_pHostBlockData = new float[totalNumBlocks()];
		m_pHostSamplesData = new unsigned int[totalNumBlocks()];
		Clear();
	}
	void Free()
	{
		CUDA_FREE(m_pLumData);
		CUDA_FREE(m_pDeviceIndexData);
		CUDA_FREE(m_pDeviceBlockData);
		CUDA_FREE(m_pDeviceSamplesData);
		delete[] m_pHostIndexData;
		delete[] m_pHostBlockData;
		delete[] m_pHostSamplesData;
	}
	float getBlockVariance(int idx) const
	{
		return m_pHostBlockData[idx];
	}
	void AddPass();
	void Clear();
	unsigned int NumBlocks() const
	{
		if (hasValidData)
			return m_uNumBlocksToLaunch;
		else return totalNumBlocks();
	}
	unsigned int totalNumBlocks() const
	{
		return ((img->getWidth() + blockSize - 1) / blockSize) * ((img->getHeight() + blockSize - 1) / blockSize);
	}
	unsigned int numBlocksRow() const
	{
		return (img->getWidth() + blockSize - 1) / blockSize;
	}
	void getBlockCoords(unsigned int idx, unsigned int& x, unsigned int& y, unsigned int& w, unsigned int& h, bool ignoreData = false) const
	{
		if (hasValidData && !ignoreData)
			idx = m_pHostIndexData[idx];
		unsigned int ix = idx % numBlocksRow();
		unsigned int iy = idx / numBlocksRow();
		x = ix * blockSize;
		y = iy * blockSize;
		unsigned int x2 = (ix + 1) * blockSize, y2 = (iy + 1) * blockSize;
		w = MIN(img->getWidth(), x2) - x;
		h = MIN(img->getHeight(), y2) - y;
	}
	unsigned int mapIdx(unsigned int idx) const
	{
		return hasValidData ? m_pHostIndexData[idx] : idx;
	}

	k_BlockSampleImage getBlockImage() const
	{
		return k_BlockSampleImage(img, m_pLumData);
	}
	void DrawVariance(bool blocks) const;
	unsigned int* getNumSamplesPerBlock() const
	{
		return m_pHostSamplesData;
	}
};

/*class k_BlockSampler
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
		: m_uBlockSize(64), hasValidData(false)
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
	unsigned int NumBlocks() const
	{
		return m_uNumBlocksToLaunch;
	}
	CUDA_FUNC_IN unsigned int blockIdx(unsigned int unmappedIdx) const
	{
#ifdef ISCUDA
		return hasValidData ? m_pDeviceIndexData[unmappedIdx] : unmappedIdx;
#else
		return hasValidData ? m_pHostIndexData[unmappedIdx] : unmappedIdx;
#endif
	}
	CUDA_FUNC_IN uint2 blockCoord(unsigned int unmappedIdx) const
	{
		unsigned int idx = blockIdx(unmappedIdx);
		unsigned int x = idx % m_uBlockDimX, y = idx / m_uBlockDimX;
		return make_uint2(x, y);
	}
	CUDA_FUNC_IN uint2 blockCoord() const
	{
#ifdef ISCUDA
		unsigned int unmappedIdx = ::blockIdx.x;
#else
		unsigned int unmappedIdx = 0;
#endif
		return blockCoord(unmappedIdx);
	}
	CUDA_FUNC_IN uint2 pixelCoord() const
	{
#ifdef ISCUDA
		uint2 tc = make_uint2(threadIdx.x, threadIdx.y);
#else
		uint2 tc = make_uint2(0, 0);
#endif
		uint2 bc = blockCoord();
		return make_uint2(bc.x * 32 + tc.x, bc.y * 32 + tc.y);
	}
	CUDA_HOST dim3 blockDim() const
	{
		return dim3(m_uNumBlocksToLaunch);
	}
	CUDA_HOST dim3 threadDim() const
	{
		return dim3(32, 32, 1);
	}
	CUDA_FUNC_IN void getRect(unsigned int idx, unsigned int& x, unsigned int& y, unsigned int& w, unsigned int& h) const
	{
		w = h = m_uBlockSize;
		uint2 q = blockCoord(idx);
		x = q.x * m_uBlockSize;
		y = m_uTargetHeight - q.y * m_uBlockSize - 1 - h;
	}
};*/