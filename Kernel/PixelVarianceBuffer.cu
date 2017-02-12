#include "PixelVarianceBuffer.h"
#include "Tracer.h"
#include <Kernel/BlockSampler/IBlockSampler.h>

namespace CudaTracerLib
{

CUDA_CONST char g_BlockFlags[(4096 / BLOCK_SAMPLER_BlockSize) * (4096 / BLOCK_SAMPLER_BlockSize)];

CUDA_GLOBAL void updateVarianceBuffer(PixelVarianceBuffer buf, Image img, unsigned int numPasses, float splatScale, unsigned int numTotalBlocksX)
{
	unsigned int x = threadIdx.x + blockDim.x * blockIdx.x, y = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int b_x = x / BLOCK_SAMPLER_BlockSize, b_y = y / BLOCK_SAMPLER_BlockSize, bIdx = b_y * numTotalBlocksX + b_x;

	if (x < img.getWidth() && y < img.getHeight() && g_BlockFlags[bIdx])
	{
		buf(x, y).updateMoments(img.getPixelData(x, y), splatScale, g_BlockFlags[bIdx]);
	}
}

void PixelVarianceBuffer::AddPass(Image& img, float splatScale, const IBlockSampler* blockSampler)
{
	auto nBlocks = blockSampler->getNumTotalBlocks();
	char* blockFlags = (char*)alloca(nBlocks);
	Platform::SetMemory(blockFlags, nBlocks);
	blockSampler->IterateBlocks([&](unsigned int i, int x, int y, int bw, int bh)
	{
		blockFlags[i]++;
	});
	ThrowCudaErrors(cudaMemcpyToSymbol(g_BlockFlags, blockFlags, nBlocks));

	m_numPasses++;

	const int cBlock = 32;
	updateVarianceBuffer << <dim3(img.getWidth() / cBlock + 1, img.getHeight() / cBlock + 1), dim3(cBlock, cBlock) >> > (*this, img, m_numPasses, splatScale, blockSampler->getTotalBlocksXDim());
	m_pixelBuffer.setOnGPU();
}

}