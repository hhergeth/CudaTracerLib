#pragma once

#include "Grid.h"
#include <CudaMemoryManager.h>

namespace CudaTracerLib {

template<typename T, typename HASHER> class SpatialGrid
{
protected:
	HashGrid_Reg hashMap;
public:
	CUDA_FUNC_IN SpatialGrid()
	{
	}

	CUDA_FUNC_IN const HashGrid_Reg& getHashGrid() const
	{
		return hashMap;
	}

	template<unsigned int MAX_ENTRIES_PER_CELL = UINT_MAX, typename CLB> CUDA_FUNC_IN void ForAll(const Vec3u& min, const Vec3u& max, const CLB& clb)
	{
		ForAllCells(min, max, [&](const Vec3u& cell_idx)
		{
			((HASHER*)this)->ForAllCellEntries<MAX_ENTRIES_PER_CELL>(cell_idx, clb);
		});
	}

	template<unsigned int MAX_ENTRIES_PER_CELL = UINT_MAX, typename CLB> CUDA_FUNC_IN void ForAll(const Vec3f& p, const CLB& clb)
	{
		((HASHER*)this)->ForAllCellEntries<MAX_ENTRIES_PER_CELL>(hashMap.Transform(p), clb);
	}

	template<unsigned int MAX_ENTRIES_PER_CELL = UINT_MAX, typename CLB> CUDA_FUNC_IN void ForAll(const Vec3f& min, const Vec3f& max, const CLB& clb)
	{
		ForAll<MAX_ENTRIES_PER_CELL>(hashMap.Transform(min), hashMap.Transform(max), clb);
	}

	template<typename CLB> CUDA_FUNC_IN void ForAllCells(const Vec3u& min_cell, const Vec3u& max_cell, const CLB& clb)
	{
		for (unsigned int ax = min_cell.x; ax <= max_cell.x; ax++)
			for (unsigned int ay = min_cell.y; ay <= max_cell.y; ay++)
				for (unsigned int az = min_cell.z; az <= max_cell.z; az++)
				{
					clb(Vec3u(ax, ay, az));
				}
	}

	template<typename CLB> CUDA_FUNC_IN void ForAllCells(const Vec3f& min, const Vec3f& max, const CLB& clb)
	{
		ForAllCells(hashMap.Transform(min), hashMap.Transform(max), clb);
	}

	template<typename CLB> CUDA_FUNC_IN void ForAllCells(const CLB& clb)
	{
		ForAllCells(Vec3u(0), Vec3u(hashMap.m_gridSize - 1), clb);
	}
};

//a mapping from R^3 -> T^n, ie. associating variable number of values with each point in the grid
template<typename T> class SpatialLinkedMap : public SpatialGrid<T, SpatialLinkedMap<T>>
{
public:
	struct linkedEntry
	{
		unsigned int nextIdx;
		T value;
	};
private:
	unsigned int numData, gridSize;
	linkedEntry* deviceData;
	unsigned int* deviceMap;
	unsigned int deviceDataIdx;
public:

	CUDA_FUNC_IN SpatialLinkedMap(){}
	SpatialLinkedMap(unsigned int gridSize, unsigned int numData)
		: numData(numData), gridSize(gridSize)
	{
		CUDA_MALLOC(&deviceData, sizeof(linkedEntry) * numData);
		CUDA_MALLOC(&deviceMap, sizeof(unsigned int) * gridSize * gridSize * gridSize);
	}
	
	void Free()
	{
		CUDA_FREE(deviceData);
		CUDA_FREE(deviceMap);
	}

	void SetSceneDimensions(const AABB& box)
	{
		hashMap = HashGrid_Reg(box, gridSize);
	}

	void ResetBuffer()
	{
		deviceDataIdx = 0;
		ThrowCudaErrors(cudaMemset(deviceMap, -1, sizeof(unsigned int) * gridSize * gridSize * gridSize));
	}

	unsigned int getNumEntries() const
	{
		return numData;
	}

	unsigned int getNumStoredEntries() const
	{
		return deviceDataIdx;
	}

	void PrepareForUse() {}

	CUDA_FUNC_IN bool isFull() const
	{
		return deviceDataIdx >= numData;
	}

	CUDA_ONLY_FUNC void store(const Vec3u& p, const T& v, unsigned int data_idx)
	{
		unsigned int map_idx = hashMap.Hash(p);
#ifdef ISCUDA
		unsigned int old_idx = atomicExch(deviceMap + map_idx, data_idx);
#else
		unsigned int old_idx = Platform::Exchange(deviceMap + map_idx, data_idx);
#endif
		//copy actual data
		deviceData[data_idx].value = v;
		deviceData[data_idx].nextIdx = old_idx;
	}

#ifdef __CUDACC__
	CUDA_ONLY_FUNC bool store(const Vec3u& p, const T& v)
	{
		//build linked list and spatial map
		unsigned int data_idx = atomicInc(&deviceDataIdx, (unsigned int)-1);
		if (data_idx >= numData)
			return false;
		unsigned int map_idx = hashMap.Hash(p);
#ifdef ISCUDA
		unsigned int old_idx = atomicExch(deviceMap + map_idx, data_idx);
#else
		unsigned int old_idx = Platform::Exchange(deviceMap + map_idx, data_idx);
#endif
		//copy actual data
		deviceData[data_idx].value = v;
		deviceData[data_idx].nextIdx = old_idx;
		return true;
	}

	CUDA_ONLY_FUNC bool store(const Vec3f& p, const T& v)
	{
		return store(hashMap.Transform(p), v);
	}

	CUDA_ONLY_FUNC unsigned int allocStorage(unsigned int n)
	{
		unsigned int idx = atomicAdd(&deviceDataIdx, n);
		return idx;
	}
#endif

	template<unsigned int MAX_ENTRIES_PER_CELL = UINT_MAX, typename CLB> CUDA_FUNC_IN void ForAllCellEntries(const Vec3u& p, const CLB& clb)
	{
		unsigned int i0 = hashMap.Hash(p), i = deviceMap[i0], N = 0;
		while (i != UINT_MAX && i != 0xffffff && N++ < MAX_ENTRIES_PER_CELL)
		{
			clb(i, deviceData[i].value);
			i = deviceData[i].nextIdx;
		}
	}

	CUDA_FUNC_IN const T& operator()(unsigned int idx) const
	{
		return deviceData[idx].value;
	}

	CUDA_FUNC_IN T& operator()(unsigned int idx)
	{
		return deviceData[idx].value;
	}

	linkedEntry* getDeviceData() { return deviceData; }
	unsigned int* getDeviceGrid() { return deviceMap; }
};

//a mapping from R^3 -> T, ie. associating one element with each point in the grid
template<typename T> struct SpatialSet
{
	unsigned int gridSize;
	T* deviceData;
	HashGrid_Reg hashMap;
public:
	SpatialSet(){}
	SpatialSet(unsigned int gridSize)
		: gridSize(gridSize)
	{
		CUDA_MALLOC(&deviceData, sizeof(T) * gridSize * gridSize * gridSize);
	}
	
	void Free()
	{
		CUDA_FREE(deviceData);
	}

	void SetSceneDimensions(const AABB& box)
	{
		hashMap = HashGrid_Reg(box, gridSize);
	}

	void ResetBuffer()
	{
		ThrowCudaErrors(cudaMemset(deviceData, 0, sizeof(T) * gridSize * gridSize * gridSize));
	}

	CUDA_FUNC_IN const T& operator()(const Vec3f& p) const
	{
		return deviceData[hashMap.Hash(p)].value;
	}

	CUDA_FUNC_IN T& operator()(const Vec3f& p)
	{
		return deviceData[hashMap.Hash(p)];
	}

	CUDA_FUNC_IN const T& operator()(unsigned int idx) const
	{
		return deviceData[idx];
	}

	CUDA_FUNC_IN T& operator()(unsigned int idx)
	{
		return deviceData[idx];
	}

	CUDA_FUNC_IN unsigned int NumEntries()
	{
		return gridSize * gridSize * gridSize;
	}
};

}
