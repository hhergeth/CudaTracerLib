#pragma once

#include "HashGrid.h"
#include <Base/Timer.h>

namespace CudaTracerLib {

template<typename T, typename HASHER> class SpatialGridBase
{
protected:
	HashGrid_Reg hashMap;
public:
	virtual ~SpatialGridBase()
	{

	}
	CUDA_FUNC_IN const HashGrid_Reg& getHashGrid() const
	{
		return hashMap;
	}

	template<typename CLB> CUDA_FUNC_IN void ForAllCells(const Vec3u& min_cell, const Vec3u& max_cell, CLB clb)
	{
		Vec3u a = min(min_cell, hashMap.m_gridDim - Vec3u(1)), b = min(max_cell, hashMap.m_gridDim - Vec3u(1));
		for (unsigned int ax = a.x; ax <= b.x; ax++)
			for (unsigned int ay = a.y; ay <= b.y; ay++)
				for (unsigned int az = a.z; az <= b.z; az++)
				{
					clb(Vec3u(ax, ay, az));
				}
	}

	template<typename CLB> CUDA_FUNC_IN void ForAllCells(const Vec3f& min, const Vec3f& max, CLB clb)
	{
		ForAllCells(hashMap.Transform(min), hashMap.Transform(max), clb);
	}

	template<typename CLB> CUDA_FUNC_IN void ForAllCells(CLB clb)
	{
		ForAllCells(Vec3u(0), hashMap.m_gridDim - Vec3u(1), clb);
	}
};

}
