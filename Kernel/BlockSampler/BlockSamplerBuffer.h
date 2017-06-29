#pragma once
#include "IBlockSampler.h"
#include <Base/SynchronizedBuffer.h>
#include "IBlockSampler_device.h"

namespace CudaTracerLib {

class BlockSamplerBuffer
{
private:
	SynchronizedBuffer<signed char> m_buffer;
	unsigned int n_x_dim;
public:
	BlockSamplerBuffer()
		: m_buffer(1)
	{

	}

	void Free()
	{
		m_buffer.Free();
	}

	virtual void Resize(unsigned int w, unsigned int h)
	{
		unsigned int a = (w + BLOCK_SAMPLER_BlockSize - 1) / BLOCK_SAMPLER_BlockSize, b = (h + BLOCK_SAMPLER_BlockSize - 1) / BLOCK_SAMPLER_BlockSize;
		m_buffer.Resize(a * b);
	}

	virtual void Update(IBlockSampler* sampler)
	{
		n_x_dim = sampler->getTotalBlocksXDim();
		if (!sampler)
		{
			m_buffer.Memset((unsigned char)1);
			return;
		}

		m_buffer.Memset((unsigned char)0);
		sampler->IterateBlocks([&](unsigned int idx, int x, int y, int w, int h)
		{
			unsigned int a = x / BLOCK_SAMPLER_BlockSize, b = y / BLOCK_SAMPLER_BlockSize;
			m_buffer[b * n_x_dim + a]++;
		});
		m_buffer.setOnCPU();
		m_buffer.Synchronize();
	}

	CUDA_FUNC_IN unsigned int getNumSamplesPerPixel(unsigned int x, unsigned int y)
	{
		unsigned int a = x / BLOCK_SAMPLER_BlockSize, b = y / BLOCK_SAMPLER_BlockSize;
		return m_buffer[b * n_x_dim + a];
	}
};

}