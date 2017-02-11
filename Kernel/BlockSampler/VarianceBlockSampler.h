#pragma once

#include "IBlockSampler_device.h"
#include "IBlockSampler.h"
#include <Engine/Image.h>
#include <Engine/SynchronizedBuffer.h>
#include <algorithm>

namespace CudaTracerLib
{

class VarianceBlockSampler : public IUserPreferenceSampler
{
public:
	struct TmpBlockInfo
	{
		//the average variance of the pixel estimator (single channel luminance) for a block
		float BLOCK_VAR_I;
		unsigned int NUM_PIXELS_VAR;

		float BLOCK_E_I;
		float BLOCK_E_I2;
		unsigned int NUM_PIXELS_E;

		CUDA_FUNC_IN float getWeight()
		{
			const float lambda = 0.5f;

			float E_I = BLOCK_E_I / NUM_PIXELS_E;

			//average standard deviation of pixel estimators
			float I_std_dev = math::sqrt(BLOCK_VAR_I / NUM_PIXELS_VAR);
			float w1 = I_std_dev / E_I;//normalized std dev

			//variance of pixel colors in block
			float I_var = BLOCK_E_I2 / NUM_PIXELS_E - math::sqr(E_I);
			float w2 = math::sqrt(I_var) / E_I;//normalized variance

			return lambda * w1 + (1 - lambda) * w2;
		}
	};
private:
	std::vector<int> m_indices;
	SynchronizedBuffer<TmpBlockInfo> m_blockInfo;
	unsigned int m_uPassesDone;
public:
	VarianceBlockSampler(unsigned int w, unsigned int h)
		: IUserPreferenceSampler(w, h), m_blockInfo(getNumTotalBlocks())
	{
		int n(0);
		m_indices.resize(getNumTotalBlocks());
		std::generate(std::begin(m_indices), std::end(m_indices), [&] { return n++; });
	}

	virtual void Free()
	{
		m_blockInfo.Free();
	}

	virtual IBlockSampler* CreateForSize(unsigned int w, unsigned int h)
	{
		return new VarianceBlockSampler(w, h);
	}

	virtual void StartNewRendering(DynamicScene* a_Scene, Image* img);

	virtual void AddPass(Image* img, TracerBase* tracer, const PixelVarianceBuffer& varBuffer);

	virtual void IterateBlocks(iterate_blocks_clb_t clb);
};

}