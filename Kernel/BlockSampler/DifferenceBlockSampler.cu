#include "DifferenceBlockSampler.h"
#include <Kernel/Tracer.h>

namespace CudaTracerLib
{

CUDA_GLOBAL void updateInfo(DifferenceBlockSampler::blockInfo* a_pTmpBlockInfoDevice, IBlockSampler::BlockInfo* a_pPersBlockInfoDevice, PixelVarianceBuffer varBuffer, Image img, unsigned int numTotalBlocksX, int numPasses)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int b_x = x / BLOCK_SAMPLER_BlockSize, b_y = y / BLOCK_SAMPLER_BlockSize, bIdx = b_y * numTotalBlocksX + b_x;

	if (x < img.getWidth() && y < img.getHeight())
	{
		if (numPasses == 1)
			return;

		float e_p = varBuffer(x, y).computeError();

		auto& bInfo = a_pTmpBlockInfoDevice[bIdx];
		atomicAdd(&bInfo.sum_e, e_p);
		atomicInc(&bInfo.n_pixels, 0xffffffff);
	}
}

void DifferenceBlockSampler::StartNewRendering(DynamicScene* a_Scene, Image* img)
{
	IUserPreferenceSampler::StartNewRendering(a_Scene, img);
	m_uPassesDone = 0;
}

void DifferenceBlockSampler::AddPass(Image* img, TracerBase* tracer, const PixelVarianceBuffer& varBuffer)
{
	if (m_uPassesDone++ == 0)
		return;

	const int cBlock = 16;
	int nx = (img->getWidth() + cBlock - 1) / cBlock, ny = (img->getHeight() + cBlock - 1) / cBlock;

	blockBuffer.Memset(0);
	updateInfo << <dim3(nx, ny), dim3(cBlock, cBlock) >> > (blockBuffer.getDevicePtr(), m_sBlockInfo.getDevicePtr(), varBuffer, *img, getTotalBlocksXDim(), m_uPassesDone);
	ThrowCudaErrors(cudaThreadSynchronize());
	blockBuffer.setOnGPU();
	blockBuffer.Synchronize();

	std::sort(std::begin(m_indices), std::end(m_indices), [&](int i1, int i2)
	{
		return blockBuffer[i1].error(img->getWidth(), img->getHeight()) * math::sqr(m_userWeights[i1]) > blockBuffer[i2].error(img->getWidth(), img->getHeight()) * math::sqr(m_userWeights[i2]);
	});

	IUserPreferenceSampler::AddPass(img, tracer, varBuffer);
}

void DifferenceBlockSampler::IterateBlocks(iterate_blocks_clb_t clb) const
{
	if (m_uPassesDone < 10)
		IterateAllBlocksUniform(clb);
	else MixedBlockIterate(m_indices, clb, m_uPassesDone, m_settings.getValue(KEY_FractionDeterministic()), m_settings.getValue(KEY_FractionWeighted()));
}

}