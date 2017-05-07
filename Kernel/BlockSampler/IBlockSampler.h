#pragma once
#include <Engine/Image.h>
#include <functional>
#include <vector>
#include <algorithm>
#include <Kernel/PixelVarianceBuffer.h>
#include <vector>

namespace CudaTracerLib {

class DynamicScene;
class TracerBase;

//args = flattened_idx, x, y, bw, bh
using iterate_blocks_clb_t = std::function<void(unsigned int, int, int, int, int)>;

class IBlockSampler
{
public:
	struct BlockInfo
	{
		unsigned int passesDone;
		BlockInfo()
			: passesDone(0)
		{

		}
	};
protected:
	DynamicScene* m_pScene;
	unsigned int xResolution, yResolution;
	SynchronizedBuffer<BlockInfo> m_sBlockInfo;
	IBlockSampler(unsigned int xResolution, unsigned int yResolution)
		: xResolution(xResolution), yResolution(yResolution), m_sBlockInfo(getNumTotalBlocks())
	{

	}
public:
	virtual ~IBlockSampler()
	{

	}

	virtual void Free() = 0;

	//Creates a new instance of this sampler for the specific image size
	virtual IBlockSampler* CreateForSize(unsigned int w, unsigned int h) = 0;

	virtual void StartNewRendering(DynamicScene* a_Scene, Image* img)
	{
		m_pScene = a_Scene;
		m_sBlockInfo.Memset(BlockInfo());
	}

	virtual void AddPass(Image* img, TracerBase* tracer, const PixelVarianceBuffer& varBuffer)
	{
		IterateBlocks([&](unsigned int f_idx, int bx, int by, int bw, int bh)
		{
			m_sBlockInfo[f_idx].passesDone++;
		});
		m_sBlockInfo.setOnCPU();
		m_sBlockInfo.Synchronize();
	}

	virtual void IterateBlocks(iterate_blocks_clb_t clb) const = 0;

	int getNumTotalBlocks() const
	{
		return getTotalBlocksXDim() * getTotalBlocksYDim();
	}

	int getTotalBlocksXDim() const
	{
		return (xResolution + BLOCK_SAMPLER_BlockSize - 1) / BLOCK_SAMPLER_BlockSize;
	}

	int getTotalBlocksYDim() const
	{
		return (yResolution + BLOCK_SAMPLER_BlockSize - 1) / BLOCK_SAMPLER_BlockSize;
	}

	int getFlattenedIdx(int block_x, int block_y) const
	{
		return getTotalBlocksXDim() * block_y + block_x;
	}

	void getIdxComponents(int flattened_idx, int& block_x, int& block_y) const
	{
		block_x = flattened_idx % getTotalBlocksXDim();
		block_y = flattened_idx / getTotalBlocksXDim();
	}

	void getBlockRect(int block_x, int block_y, int& x, int& y, int& bw, int& bh) const
	{
		//compute bounds of block
		x = block_x * BLOCK_SAMPLER_BlockSize;
		y = block_y * BLOCK_SAMPLER_BlockSize;
		int x2 = (block_x + 1) * BLOCK_SAMPLER_BlockSize, y2 = (block_y + 1) * BLOCK_SAMPLER_BlockSize;
		bw = min(int(xResolution), x2) - x;
		bh = min(int(yResolution), y2) - y;
	}

	void IterateAllBlocksUniform(iterate_blocks_clb_t clb) const
	{
		unsigned int block_idx = 0;
		int nx = (xResolution + BLOCK_SAMPLER_BlockSize - 1) / BLOCK_SAMPLER_BlockSize,
			ny = (yResolution + BLOCK_SAMPLER_BlockSize - 1) / BLOCK_SAMPLER_BlockSize;
		for (int ix = 0; ix < nx; ix++)
			for (int iy = 0; iy < ny; iy++)
			{
				int x, y, bw, bh;
				getBlockRect(ix, iy, x, y, bw, bh);
				clb(block_idx++, x, y, bw, bh);
			}
	}

	const BlockInfo* getBlockInfo() const
	{
		return &m_sBlockInfo[0];
	}

protected:
	//splits between sampling blocks deterministically and based on the weighting scheme present in indices
	void MixedBlockIterate(const std::vector<int>& indices, iterate_blocks_clb_t clb, int passCounter, int num_deterministic = 2) const
	{
		for (int i = 0; i < getNumTotalBlocks() / 4; i++)
		{
			auto flattened_idx = indices[i];
			int block_x, block_y, x, y, bw, bh;
			getIdxComponents(flattened_idx, block_x, block_y);

			getBlockRect(block_x, block_y, x, y, bw, bh);

			clb(flattened_idx, x, y, bw, bh);
		}

		int start_deterministic = passCounter % num_deterministic;//deterministically sample the same number of blocks every n passes
		for (int i = start_deterministic; i < getNumTotalBlocks(); i += num_deterministic)
		{
			int block_x, block_y, x, y, bw, bh;
			getIdxComponents(i, block_x, block_y);

			getBlockRect(block_x, block_y, x, y, bw, bh);

			clb(i, x, y, bw, bh);
		}
	}
};

class IUserPreferenceSampler : public IBlockSampler
{
protected:
	std::vector<float> m_userWeights;
public:
	IUserPreferenceSampler(unsigned int w, unsigned int h)
		: IBlockSampler(w, h), m_userWeights(getNumTotalBlocks(), 1.0f)
	{
	}

	virtual ~IUserPreferenceSampler()
	{

	}

	float getWeight(int block_x, int block_y)
	{
		return m_userWeights[getFlattenedIdx(block_x, block_y)];
	}

	void setWeight(int block_x, int block_y, float val)
	{
		m_userWeights[getFlattenedIdx(block_x, block_y)] = val;
	}
};

}