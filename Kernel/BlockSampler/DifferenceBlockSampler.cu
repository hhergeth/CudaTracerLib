#include "DifferenceBlockSampler.h"
#include <Kernel/Tracer.h>

namespace CudaTracerLib
{

CUDA_GLOBAL void updateInfo(DifferenceBlockSampler::blockInfo* a_pTmpBlockInfoDevice, IBlockSampler::BlockInfo* a_pPersBlockInfoDevice, Spectrum* lastAccumBuffer, Spectrum* otherAccumBuffer, Image img, unsigned int numTotalBlocksX, int numPasses)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int b_x = x / BLOCK_SAMPLER_BlockSize, b_y = y / BLOCK_SAMPLER_BlockSize, bIdx = b_y * numTotalBlocksX + b_x;

	if (x < img.getWidth() && y < img.getHeight())
	{
		Spectrum sum_new;
		sum_new.fromLinearRGB(img.getPixelData(x, y).rgb[0], img.getPixelData(x, y).rgb[1], img.getPixelData(x, y).rgb[2]);
		if (numPasses % 2 == 0)
		{
			Spectrum sample = sum_new - lastAccumBuffer[y * img.getWidth() + x];
			lastAccumBuffer[y * img.getWidth() + x] = sum_new;
			otherAccumBuffer[y * img.getWidth() + x] += sample;
		}

		if (numPasses == 1)
			return;

		Spectrum I = sum_new / numPasses;
		Spectrum A = otherAccumBuffer[y * img.getWidth() + x] / float(numPasses / 2);
		Vec3f rgb, I_rgb;
		Spectrum(I - A).toLinearRGB(rgb[0], rgb[1], rgb[2]);
		I.toLinearRGB(I_rgb[0], I_rgb[1], I_rgb[2]);
		float e_p = I.isZero() || I.isNaN() || A.isNaN() ? 0.0f : rgb.abs().sum() / max(I_rgb.length(), 1e-2f);

		auto& bInfo = a_pTmpBlockInfoDevice[bIdx];
		atomicAdd(&bInfo.sum_e, e_p);
		atomicInc(&bInfo.n_pixels, 0xffffffff);
	}
}

void DifferenceBlockSampler::StartNewRendering(DynamicScene* a_Scene, Image* img)
{
	IUserPreferenceSampler::StartNewRendering(a_Scene, img);
	m_uPassesDone = 0;
	cudaMemset(lastAccumBuffer, 0, sizeof(Spectrum) * img->getWidth() * img->getHeight());
	cudaMemset(halfAccumBuffer, 0, sizeof(Spectrum) * img->getWidth() * img->getHeight());
}

void DifferenceBlockSampler::AddPass(Image* img, TracerBase* tracer, const PixelVarianceBuffer& varBuffer)
{
	m_uPassesDone++;

	const int cBlock = 16;
	int nx = (img->getWidth() + cBlock - 1) / cBlock, ny = (img->getHeight() + cBlock - 1) / cBlock;

	blockBuffer.Memset(0);
	updateInfo << <dim3(nx, ny), dim3(cBlock, cBlock) >> > (blockBuffer.getDevicePtr(), m_sBlockInfo.getDevicePtr(), lastAccumBuffer, halfAccumBuffer, *img, getTotalBlocksXDim(), m_uPassesDone);
	ThrowCudaErrors(cudaThreadSynchronize());
	blockBuffer.setOnGPU();
	blockBuffer.Synchronize();

	if(m_uPassesDone > 1)
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
	else MixedBlockIterate(m_indices, clb, m_uPassesDone);
}

}