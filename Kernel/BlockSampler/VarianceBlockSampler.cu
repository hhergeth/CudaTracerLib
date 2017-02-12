#include "VarianceBlockSampler.h"
#include <Kernel/Tracer.h>

namespace CudaTracerLib
{

CUDA_GLOBAL void updateInfo(VarianceBlockSampler::TmpBlockInfo* a_pTmpBlockInfoDevice, IBlockSampler::BlockInfo* a_pPersBlockInfoDevice, const PixelVarianceBuffer varBuffer, Image img, float splatScale, unsigned int numTotalBlocksX)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int b_x = x / BLOCK_SAMPLER_BlockSize, b_y = y / BLOCK_SAMPLER_BlockSize, bIdx = b_y * numTotalBlocksX + b_x;

	if (x < img.getWidth() && y < img.getHeight())
	{
		auto num_passes_block = a_pPersBlockInfoDevice[bIdx].passesDone;

		auto I_N = img.getPixelData(x, y).toSpectrum(splatScale);

		auto pInfo = varBuffer(x, y);
		auto var = pInfo.I.Var(pInfo.numSamples);
		auto e = pInfo.I.E(pInfo.numSamples);

		auto& bInfo = a_pTmpBlockInfoDevice[bIdx];
		if (var >= 0 && !math::IsNaN(var))
		{
			atomicAdd(&bInfo.BLOCK_VAR_I, var);
			atomicInc(&bInfo.NUM_PIXELS_VAR, 0xffffffff);
		}
		atomicAdd(&bInfo.BLOCK_E_I, e);
		atomicAdd(&bInfo.BLOCK_E_I2, e * e);
		atomicInc(&bInfo.NUM_PIXELS_E, 0xffffffff);
	}
}

void VarianceBlockSampler::StartNewRendering(DynamicScene* a_Scene, Image* img)
{
	IUserPreferenceSampler::StartNewRendering(a_Scene, img);
	m_uPassesDone = 0;
}

void VarianceBlockSampler::AddPass(Image* img, TracerBase* tracer, const PixelVarianceBuffer& varBuffer)
{
	m_uPassesDone++;

	const int cBlock = 32;
	int nx = (img->getWidth() + cBlock - 1) / cBlock, ny = (img->getHeight() + cBlock - 1) / cBlock;

	m_blockInfo.Memset(0);
	updateInfo << <dim3(nx, ny), dim3(cBlock, cBlock) >> > (m_blockInfo.getDevicePtr(), m_sBlockInfo.getDevicePtr(), varBuffer, *img, tracer->getSplatScale(), getTotalBlocksXDim());
	m_blockInfo.setOnGPU();
	m_blockInfo.Synchronize();

	float min_block = FLT_MAX, max_block = -FLT_MAX;
	float min_est = FLT_MAX, max_est = -FLT_MAX;
	for (unsigned int i = 0; i < m_blockInfo.getLength(); i++)
	{
		auto& b = m_blockInfo[i];
		float est_var = b.get_w1();
		float block_var = b.get_w2();
		min_block = min(min_block, block_var); max_block = max(max_block, block_var);
		min_est = min(min_est, est_var); max_est = max(max_est, est_var);
	}

	std::sort(std::begin(m_indices), std::end(m_indices), [&](int i1, int i2)
	{
		return m_blockInfo[i1].getWeight(min_block, max_block, min_est, max_est) * math::sqr(m_userWeights[i1]) > m_blockInfo[i2].getWeight(min_block, max_block, min_est, max_est) * math::sqr(m_userWeights[i2]);
	});

	IUserPreferenceSampler::AddPass(img, tracer, varBuffer);
}

void VarianceBlockSampler::IterateBlocks(iterate_blocks_clb_t clb)
{
	if(m_uPassesDone < 10)
		IterateAllBlocksUniform(clb);
	else
	{
		for (int i = 0; i < getNumTotalBlocks() / 4; i++)
		{
			auto flattened_idx = m_indices[i];
			int block_x, block_y, x, y, bw, bh;
			getIdxComponents(flattened_idx, block_x, block_y);

			getBlockRect(block_x, block_y, x, y, bw, bh);

			clb(flattened_idx, x, y, bw, bh);
		}

		int num_deterministic = 2;
		int start_deterministic = m_uPassesDone % num_deterministic;//deterministically sample the same number of blocks every n passes
		for (int i = start_deterministic; i < getNumTotalBlocks(); i += num_deterministic)
		{
			int block_x, block_y, x, y, bw, bh;
			getIdxComponents(i, block_x, block_y);

			getBlockRect(block_x, block_y, x, y, bw, bh);

			clb(i, x, y, bw, bh);
		}
	}
}

}