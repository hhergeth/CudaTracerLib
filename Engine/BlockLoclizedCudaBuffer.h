#pragma once

#include <Defines.h>

namespace CudaTracerLib {
	
	template<typename T, int BLOCK_SIZE = BLOCK_SAMPLER_BlockSize> class BlockLoclizedCudaBuffer
	{
		//this is a grid based storage layout; grids are stored row first; pixels in grid also
		struct BlockData
		{
			T data[BLOCK_SIZE * BLOCK_SIZE];
		};
		BlockData* m_hostData;//array of m_width_blocks * m_height_blocks blocks
		BlockData* m_deviceData;//pointer to a single block in device memory
		unsigned int m_width_pixels, m_height_pixels;
		unsigned int m_width_blocks, m_height_blocks;
		unsigned int m_original_width_pixels, m_original_height_pixels;
		int current_block_x_pixels, current_block_y_pixels;
	public:
		BlockLoclizedCudaBuffer(unsigned int w, unsigned int h)
			: m_original_width_pixels(w), m_original_height_pixels(h), current_block_x_pixels(-1), current_block_y_pixels(-1)
		{
			static_assert(sizeof(BlockData) == sizeof(T) * BLOCK_SIZE * BLOCK_SIZE, "Please verify that there is no padding!");
			auto rem_w = w % BLOCK_SIZE, rem_h = h % BLOCK_SIZE;
			m_width_pixels = rem_w ? w + BLOCK_SIZE - rem_w : w;
			m_height_pixels = rem_h ? h + BLOCK_SIZE - rem_h : h;
			m_width_blocks = m_width_pixels / BLOCK_SIZE;
			m_height_blocks = m_height_pixels / BLOCK_SIZE;
			m_hostData = new BlockData[m_width_blocks * m_height_blocks];
			CUDA_MALLOC(&m_deviceData, sizeof(BlockData));
		}
		virtual ~BlockLoclizedCudaBuffer()
		{

		}
		virtual void Free()
		{
			delete[] m_hostData;
			CUDA_FREE(m_deviceData);
		}
		//x, y in pixels
		virtual void StartBlock(int x, int y, bool copyData = true)
		{
			current_block_x_pixels = x;
			current_block_y_pixels = y;
			if ((x % BLOCK_SIZE) != 0 || (y % BLOCK_SIZE) != 0)
				throw std::runtime_error("Cannot copy intersection of multiple blocks!");
			if(copyData)
				CUDA_MEMCPY_TO_DEVICE(m_deviceData, m_hostData + (y / BLOCK_SIZE) * m_width_blocks + (x / BLOCK_SIZE), sizeof(BlockData));
		}
		virtual void EndBlock()
		{
			if ((current_block_x_pixels % BLOCK_SIZE) != 0 || (current_block_y_pixels % BLOCK_SIZE) != 0 || 
				 current_block_x_pixels < 0 || current_block_y_pixels < 0)
				throw std::runtime_error("Cannot copy intersection of multiple blocks!");
			CUDA_MEMCPY_TO_HOST(m_hostData + (current_block_y_pixels / BLOCK_SIZE) * m_width_blocks + (current_block_x_pixels / BLOCK_SIZE), m_deviceData, sizeof(BlockData));
			current_block_x_pixels = current_block_y_pixels = -1;
		}
		virtual void CopyLinearBuffer(T* linearBuffer, bool copyTo) const
		{
			auto rem_w = m_original_width_pixels % BLOCK_SIZE, rem_h = m_original_height_pixels % BLOCK_SIZE;
			for (unsigned int block_idx = 0; block_idx < m_width_blocks * m_height_blocks; block_idx++)
			{
				unsigned int block_i = block_idx % m_width_blocks, block_j = block_idx / m_width_blocks;
				bool last_block_in_row = block_i == m_width_blocks - 1, last_block_in_col = block_j == m_height_blocks - 1;
				unsigned int num_rows_to_copy = last_block_in_col ? (rem_h ? rem_h : BLOCK_SIZE) : BLOCK_SIZE;
				unsigned int num_pixels_to_copy = last_block_in_row ? (rem_w ? rem_w : BLOCK_SIZE) : BLOCK_SIZE;

				for (unsigned int i = 0; i < num_rows_to_copy; i++)
				{
					auto x = block_i * BLOCK_SIZE, y = block_j * BLOCK_SIZE + i;
					auto* linBuf = linearBuffer + y * m_original_width_pixels + x;
					auto* blockBuf = m_hostData[block_idx].data + i * BLOCK_SIZE;
					memcpy(copyTo ? linBuf : blockBuf, copyTo ? blockBuf : linBuf, num_pixels_to_copy * sizeof(T));
				}
			}
		}
		CUDA_FUNC_IN const T& operator()(int x, int y) const
		{
			return ((BlockLoclizedCudaBuffer<T, BLOCK_SIZE>*)this)->operator()(x, y);
		}
		CUDA_FUNC_IN T& operator()(int x, int y)
		{
#ifdef ISCUDA
			CTL_ASSERT(x >= current_block_x_pixels && x < current_block_x_pixels + BLOCK_SIZE && y >= current_block_y_pixels && y < current_block_y_pixels + BLOCK_SIZE);
			x = math::clamp(x - current_block_x_pixels, 0, BLOCK_SIZE - 1);
			y = math::clamp(y - current_block_y_pixels, 0, BLOCK_SIZE - 1);
			return m_deviceData->data[y * BLOCK_SIZE + x];
#else
			auto& block = m_hostData[(y / BLOCK_SIZE) * m_width_blocks + x / BLOCK_SIZE];
			return block.data[(y % BLOCK_SIZE) * BLOCK_SIZE + x % BLOCK_SIZE];
#endif
		}
	};

}