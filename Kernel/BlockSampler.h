#pragma once

#include <MathTypes.h>
#include "BlockSampler_device.h"

namespace CudaTracerLib {

struct SamplerpixelData
{
	float E, E2;
	float lastVar;
	unsigned int n;
	unsigned int flag;
};

class Image;
class BlockSampler : public IBlockSampler
{
	SamplerpixelData* m_pLumData;
	Image* img;

	unsigned int m_uPassesDone;
	unsigned int* m_pDeviceIndexData;
	float* m_pDeviceBlockData;
	float* m_pHostBlockData;
	unsigned int* m_pHostIndexData;
	unsigned int m_uNumBlocksToLaunch;
	bool hasValidData;
	unsigned int* m_pDeviceSamplesData;
	unsigned int* m_pHostSamplesData;
	unsigned int* m_pDeviceContribPixels;
	unsigned int* m_pHostContribPixels;
	float* m_pHostWeight;
	float* m_pDeviceWeight;
public:
	bool m_bIsActive;

	BlockSampler(Image* img);
	virtual ~BlockSampler()
	{
		Free();
	}
	virtual void Free();
	float getBlockVariance(int idx) const
	{
		return m_pHostBlockData[idx];
	}
	virtual void AddPass();
	virtual void Clear();
	virtual unsigned int NumBlocks() const
	{
		if (hasValidData)
			return m_uNumBlocksToLaunch;
		else return totalNumBlocks();
	}
	unsigned int totalNumBlocks() const
	{
		return ((img->getWidth() + BLOCK_SAMPLER_BlockSize - 1) / BLOCK_SAMPLER_BlockSize) * ((img->getHeight() + BLOCK_SAMPLER_BlockSize - 1) / BLOCK_SAMPLER_BlockSize);
	}
	unsigned int numBlocksRow() const
	{
		return (img->getWidth() + BLOCK_SAMPLER_BlockSize - 1) / BLOCK_SAMPLER_BlockSize;
	}
	virtual void getBlockCoords(unsigned int idx, unsigned int& x, unsigned int& y, unsigned int& w, unsigned int& h, bool ignoreData = false) const
	{
		if (hasValidData && !ignoreData)
			idx = m_pHostIndexData[idx];
		unsigned int ix = idx % numBlocksRow();
		unsigned int iy = idx / numBlocksRow();
		x = ix * BLOCK_SAMPLER_BlockSize;
		y = iy * BLOCK_SAMPLER_BlockSize;
		unsigned int x2 = (ix + 1) * BLOCK_SAMPLER_BlockSize, y2 = (iy + 1) * BLOCK_SAMPLER_BlockSize;
		w = min(img->getWidth(), x2) - x;
		h = min(img->getHeight(), y2) - y;
	}
	unsigned int mapIdx(unsigned int idx) const
	{
		return hasValidData ? m_pHostIndexData[idx] : idx;
	}

	virtual BlockSampleImage getBlockImage() const
	{
		return BlockSampleImage(img, m_pLumData);
	}
	void DrawVariance(bool blocks) const;
	unsigned int* getNumSamplesPerBlock() const
	{
		return m_pHostSamplesData;
	}
	float& accessWeight(unsigned int idx)
	{
		return m_pHostWeight[idx];
	}
};

}