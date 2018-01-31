#pragma once

#include "IBlockSampler_device.h"
#include "IBlockSampler.h"
#include <Engine/Image.h>
#include <Base/SynchronizedBuffer.h>

namespace CudaTracerLib {

class SelectBlockSampler : public IUserPreferenceSampler
{
public:
	SelectBlockSampler(unsigned int w, unsigned int h)
		: IUserPreferenceSampler(w, h)
	{
		m_userWeights = std::vector<float>(getNumTotalBlocks(), 0.0f);
	}

	virtual void Free()
	{

	}

	virtual IBlockSampler* CreateForSize(unsigned int w, unsigned int h)
	{
		return new SelectBlockSampler(w, h);
	}

	virtual void StartNewRendering(DynamicScene* a_Scene, Image* img)
	{
		IUserPreferenceSampler::StartNewRendering(a_Scene, img);
	}

	virtual void AddPass(Image* img, TracerBase* tracer, const PixelVarianceBuffer& varBuffer)
	{
		IUserPreferenceSampler::AddPass(img, tracer, varBuffer);
	}

	virtual void IterateBlocks(iterate_blocks_clb_t clb)  const
	{
		auto nX = getTotalBlocksXDim(), nY = getTotalBlocksYDim();

		bool oneSelected = false;
		for(int block_x = 0; block_x < nX; block_x++)
			for (int block_y = 0; block_y < nY; block_y++)
				if (IUserPreferenceSampler::getWeight(block_x, block_y) != 0.0f)
				{
					oneSelected = true;
					int i = getFlattenedIdx(block_x, block_y), x, y, bw, bh;

					getBlockRect(block_x, block_y, x, y, bw, bh);

					clb(i, x, y, bw, bh);
				}

		if (!oneSelected)
			std::cout << __FUNCTION__ << ":: not a single block was selected, doing nothing" << std::endl;
	}
};

}
