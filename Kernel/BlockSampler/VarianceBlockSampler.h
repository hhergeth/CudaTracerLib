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
		float BLOCK_E_I;
		unsigned int NUM_PIXELS;

		CUDA_FUNC_IN float getWeight()
		{
			return BLOCK_VAR_I / (BLOCK_E_I * NUM_PIXELS);
		}
	};
	struct PixelInfo
	{
		//the first moment of the sequence of estimators
		Vec3f E_I_N;
		//the second moment of...
		Vec3f E_I2_N;

		CUDA_FUNC_IN void updateMoments(const Spectrum& s)
		{
			Vec3f rgb;
			s.toLinearRGB(rgb.x, rgb.y, rgb.z);
			E_I_N += rgb;
			E_I2_N += math::sqr(rgb);
		}

		CUDA_FUNC_IN Vec3f getExpectedValue(unsigned int N) const
		{
			float f = 1.0f / N;
			return E_I_N * f;
		}

		CUDA_FUNC_IN Vec3f getVariance(unsigned int N) const
		{
			float f = 1.0f / N;
			return E_I2_N * f - math::sqr(E_I_N * f);
		}
	};
private:
	std::vector<int> m_indices;
	SynchronizedBuffer<TmpBlockInfo> m_blockInfo;
	PixelInfo* m_pPixelInfoDevice;
	unsigned int m_uPassesDone;
public:
	VarianceBlockSampler(unsigned int w, unsigned int h)
		: IUserPreferenceSampler(w, h), m_blockInfo(getNumTotalBlocks())
	{
		int n(0);
		m_indices.resize(getNumTotalBlocks());
		std::generate(std::begin(m_indices), std::end(m_indices), [&] { return n++; });

		CUDA_MALLOC(&m_pPixelInfoDevice, sizeof(PixelInfo) * w * h);
	}

	virtual void Free()
	{
		m_blockInfo.Free();
		if(xResolution * yResolution != 0)
			CUDA_FREE(m_pPixelInfoDevice);
	}

	virtual IBlockSampler* CreateForSize(unsigned int w, unsigned int h)
	{
		return new VarianceBlockSampler(w, h);
	}

	virtual void StartNewRendering(DynamicScene* a_Scene, Image* img);

	virtual void AddPass(Image* img, TracerBase* tracer);

	virtual void IterateBlocks(iterate_blocks_clb_t clb);
};

}