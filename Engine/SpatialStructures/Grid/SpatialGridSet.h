#pragma once

#include "SpatialGrid.h"
#include <Base/SynchronizedBuffer.h>

namespace CudaTracerLib {

//a mapping from R^3 -> T, ie. associating one element with each point in the grid
template<typename T> struct SpatialGridSet : public SpatialGridBase<T, SpatialGridSet<T>>, public ISynchronizedBufferParent
{
	typedef SpatialGridBase<T, SpatialGridSet<T>> BaseType;
	Vec3u m_gridSize;
	SynchronizedBuffer<T> m_buffer;
public:
	SpatialGridSet(const Vec3u& gridSize)
		: ISynchronizedBufferParent(m_buffer), m_gridSize(gridSize), m_buffer(m_gridSize.x * m_gridSize.y * m_gridSize.z)
	{
	}

	void SetGridDimensions(const AABB& box)
	{
		BaseType::hashMap = HashGrid_Reg(box, m_gridSize);
	}

	void ResetBuffer()
	{
		m_buffer.Memset((unsigned char)0);
	}

	CUDA_FUNC_IN unsigned int getNumCells() const
	{
		return m_gridSize.x * m_gridSize.y * m_gridSize.z;
	}

	CUDA_FUNC_IN const T& operator()(const Vec3f& p) const
	{
		return m_buffer[BaseType::getHashGrid().Hash(p)].value;
	}

	CUDA_FUNC_IN T& operator()(const Vec3f& p)
	{
		return m_buffer[BaseType::getHashGrid().Hash(p)];
	}

	CUDA_FUNC_IN const T& operator()(const Vec3u& p) const
	{
		return m_buffer[BaseType::getHashGrid().Hash(p)].value;
	}

	CUDA_FUNC_IN T& operator()(const Vec3u& p)
	{
		return m_buffer[BaseType::getHashGrid().Hash(p)];
	}

	CUDA_FUNC_IN const T& operator()(unsigned int idx) const
	{
		return m_buffer[idx];
	}

	CUDA_FUNC_IN T& operator()(unsigned int idx)
	{
		return m_buffer[idx];
	}
};

}