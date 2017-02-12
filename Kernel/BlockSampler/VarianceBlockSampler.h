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

		CUDA_FUNC_IN float get_w1()
		{
			if (NUM_PIXELS_VAR == 0)
				return 0.0f;
			//average standard deviation of pixel estimators
			float I_std_dev = math::sqrt(BLOCK_VAR_I / NUM_PIXELS_VAR);
			return I_std_dev;
		}

		CUDA_FUNC_IN float get_w2()
		{
			if (NUM_PIXELS_E == 0)
				return 0.0f;
			//variance of pixel colors in block
			float E_I = BLOCK_E_I / NUM_PIXELS_E;
			float I_std_dev = math::sqrt(BLOCK_E_I2 / NUM_PIXELS_E - math::sqr(E_I));
			return I_std_dev;
		}

		CUDA_FUNC_IN float getWeight(float min_block, float max_block, float min_est, float max_est)
		{
			const float lambda = 0.85f;

			float I_std_dev = get_w1();
			float w1 = (I_std_dev - min_est) / (max_est - min_est);//normalized std dev

			float I_var = get_w2();
			float w2 = (I_var - min_block) / (max_block - min_block);//normalized variance

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

	virtual void IterateBlocks(iterate_blocks_clb_t clb)  const;
};

}