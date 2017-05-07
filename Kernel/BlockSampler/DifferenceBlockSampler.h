#pragma once

#include "IBlockSampler_device.h"
#include "IBlockSampler.h"
#include <Engine/Image.h>
#include <Base/SynchronizedBuffer.h>

namespace CudaTracerLib
{

class DifferenceBlockSampler : public IUserPreferenceSampler
{
public:
	struct blockInfo
	{
		float sum_e;
		unsigned int n_pixels;

		float error(int w, int h) const
		{
			float r = math::sqrt(n_pixels / float(w * h));
			return r / n_pixels * sum_e;
		}
	};
private:
	Spectrum* lastAccumBuffer;
	Spectrum* halfAccumBuffer;
	SynchronizedBuffer<blockInfo> blockBuffer;
	int m_uPassesDone;
	std::vector<int> m_indices;
public:
	DifferenceBlockSampler(unsigned int w, unsigned int h)
		: IUserPreferenceSampler(w, h), blockBuffer(getNumTotalBlocks())
	{
		CUDA_MALLOC(&lastAccumBuffer, sizeof(Spectrum) * w * h);
		CUDA_MALLOC(&halfAccumBuffer, sizeof(Spectrum) * w * h);
		int n(0);
		m_indices.resize(getNumTotalBlocks());
		std::generate(std::begin(m_indices), std::end(m_indices), [&] { return n++; });
	}

	virtual void Free()
	{
		CUDA_FREE(lastAccumBuffer);
		CUDA_FREE(halfAccumBuffer);
		blockBuffer.Free();
	}

	virtual IBlockSampler* CreateForSize(unsigned int w, unsigned int h)
	{
		return new DifferenceBlockSampler(w, h);
	}

	virtual void StartNewRendering(DynamicScene* a_Scene, Image* img);

	virtual void AddPass(Image* img, TracerBase* tracer, const PixelVarianceBuffer& varBuffer);

	virtual void IterateBlocks(iterate_blocks_clb_t clb)  const;
};

}
