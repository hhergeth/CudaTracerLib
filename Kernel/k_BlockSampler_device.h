#pragma once
#include <Engine/e_Image.h>

namespace CudaTracerLib {

struct k_SamplerpixelData;

#ifdef _DEBUG
#define BLOCK_FACTOR 2
#else
#define BLOCK_FACTOR 4
#endif

#define threadsPerBlock dim3(16, 8)
#define numBlocks dim3(2 * BLOCK_FACTOR, 4 * BLOCK_FACTOR)
#define blockSize (32 * BLOCK_FACTOR)

struct k_BlockSampleImage
{
	e_Image img;
	k_SamplerpixelData* m_pLumData;
	unsigned int w;

	k_BlockSampleImage(e_Image* img, k_SamplerpixelData* lumData)
		: img(*img), m_pLumData(lumData)
	{
		unsigned int y;
		img->getExtent(w, y);
	}

	CUDA_DEVICE CUDA_HOST void Add(int x, int y, const Spectrum& c);
};

class IBlockSampler
{
public:
	virtual void Free() = 0;
	virtual void AddPass() = 0;
	virtual void Clear() = 0;
	virtual unsigned int NumBlocks() const = 0;
	virtual void getBlockCoords(unsigned int idx, unsigned int& x, unsigned int& y, unsigned int& w, unsigned int& h, bool ignoreData = false) const = 0;
	virtual k_BlockSampleImage getBlockImage() const = 0;
};

}