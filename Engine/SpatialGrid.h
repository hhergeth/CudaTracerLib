#pragma once

#include "Grid.h"
#include <CudaMemoryManager.h>
#ifdef __CUDACC__
#pragma warning (disable : 4267) 
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#pragma warning (enable : 4267)
#endif

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
			((HASHER*)this)->ForAllCellEntries<MAX_ENTRIES_PER_CELL>(cell_idx, [&](unsigned int e_idx, T& val)
			{
				clb(cell_idx, e_idx, val);
			});
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

#ifdef __CUDACC__
namespace __interal_spatialMap__
{

struct order
{
	CUDA_FUNC_IN bool operator()(const Vec2u& a, const Vec2u& b) const
	{
		return a.y < b.y;
	}
};

template<typename T, int N_PER_THREAD, int N_MAX_PER_CELL> __global__ void buildGrid(T* deviceDataSource, T* deviceDataDest, unsigned int N, Vec2u* deviceList, unsigned int* deviceGrid, unsigned int* g_DestCounter)
{
	static_assert(N_MAX_PER_CELL >= N_PER_THREAD, "A thread must be able to copy more elements than in his cell can be!");
	unsigned int startIdx = N_PER_THREAD * (blockIdx.x * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x), idx = startIdx;
	//skip indices from the prev cell
	if (idx > 0 && idx < N)
	{
		unsigned int prev_idx = deviceList[idx - 1].y;
		while (idx < N && deviceList[idx].y == prev_idx && idx - startIdx < N_PER_THREAD)
			idx++;
	}
	//copy, possibly leaving this thread's segment
	while (idx < N && idx - startIdx < N_PER_THREAD)
	{
		//count the number of elements in this cell
		unsigned int idxPast = idx + 1, cell_idx = deviceList[idx].y;
		while (idxPast < N && deviceList[idxPast].y == cell_idx)//&& idxPast - idx <= N_MAX_PER_CELL
			idxPast++;
		unsigned int tarBufferLoc = atomicAdd(g_DestCounter, idxPast - idx);
		deviceGrid[cell_idx] = tarBufferLoc;
		//copy the elements to the newly aquired location
		for (; idx < idxPast; idx++)
			deviceDataDest[tarBufferLoc++] = deviceDataSource[deviceList[idx].x];
		deviceDataDest[tarBufferLoc - 1].setFlag(true);
	}
}

}
#endif

template<typename T> class SpatialFlatMap : public SpatialGrid<T, SpatialFlatMap<T>>
{
public:
	unsigned int numData, idxData;
	T* deviceData, *deviceData2;
	unsigned int gridSize;
	unsigned int* deviceGrid;
	Vec2u* deviceList;
	unsigned int* m_deviceIdxCounter;

	T* hostData1, *hostData2;
	Vec2u* hostList;
	unsigned int* hostGrid;
	CUDA_FUNC_IN SpatialFlatMap()
	{

	}

	SpatialFlatMap(unsigned int gridSize, unsigned int numData)
		: numData(numData), gridSize(gridSize), idxData(0)
	{
		CUDA_MALLOC(&deviceData, sizeof(T) * numData);
		CUDA_MALLOC(&deviceData2, sizeof(T) * numData);
		CUDA_MALLOC(&deviceGrid, sizeof(unsigned int) * gridSize * gridSize * gridSize);
		CUDA_MALLOC(&deviceList, sizeof(Vec2u) * numData);
		CUDA_MALLOC(&m_deviceIdxCounter, sizeof(unsigned int));

		hostData1 = new T[numData];
		hostData2 = new T[numData];
		hostList = new Vec2u[numData];
		hostGrid = new unsigned int[gridSize * gridSize * gridSize];
	}

	void Free()
	{
		CUDA_FREE(deviceData);
		CUDA_FREE(deviceData2);
		CUDA_FREE(deviceGrid);
		CUDA_FREE(deviceList);
		CUDA_FREE(m_deviceIdxCounter);
		delete[] hostData1;
		delete[] hostData2;
		delete[] hostList;
		delete[] hostGrid;
	}

	void SetSceneDimensions(const AABB& box)
	{
		hashMap = HashGrid_Reg(box, gridSize);
	}

	void ResetBuffer()
	{
		idxData = 0;
	}

	void PrepareForUse()
	{
		auto& Tt = PerformanceTimer::getInstance(typeid(PPPMTracer).name());

		idxData = min(idxData, numData);

#ifndef __CUDACC__
		ThrowCudaErrors(cudaMemcpy(hostData1, deviceData, sizeof(T) * idxData, cudaMemcpyDeviceToHost));
		{
			auto bl = Tt.StartBlock("sort");
			thrust::sort(thrust::device_ptr<Vec2u>(deviceList), thrust::device_ptr<Vec2u>(deviceList + idxData), __interal_spatialMap__::order());
		}
		ThrowCudaErrors(cudaMemcpy(hostList, deviceList, sizeof(Vec2u) * idxData, cudaMemcpyDeviceToHost));
		{
			auto bl = Tt.StartBlock("reset");
			for (unsigned int idx = 0; idx < gridSize * gridSize * gridSize; idx++)
				hostGrid[idx] = UINT_MAX;
		}
		{
			auto bl = Tt.StartBlock("build");
			unsigned int i = 0;
			while (i < idxData)
			{
				unsigned int cellHash = hostList[i].y;
				hostGrid[cellHash] = i;
				while (i < idxData && hostList[i].y == cellHash)
				{
					hostData2[i] = hostData1[hostList[i].x];
					i++;
				}
				hostData2[i - 1].setFlag(true);
			}
		}
		ThrowCudaErrors(cudaMemcpy(deviceGrid, hostGrid, sizeof(unsigned int) * gridSize * gridSize * gridSize, cudaMemcpyHostToDevice));
		ThrowCudaErrors(cudaMemcpy(deviceData, hostData2, sizeof(T) * idxData, cudaMemcpyHostToDevice));
#else
		{
			auto bl = Tt.StartBlock("sort");
			thrust::sort(thrust::device_ptr<Vec2u>(deviceList), thrust::device_ptr<Vec2u>(deviceList + idxData), __interal_spatialMap__::order());
		}
		{
			auto bl = Tt.StartBlock("init");
			ThrowCudaErrors(cudaMemset(deviceGrid, UINT_MAX, gridSize * gridSize * gridSize));
		}
		{
			auto bl = Tt.StartBlock("build");
			const unsigned int N_THREAD = 10;
			CudaSetToZero(m_deviceIdxCounter, sizeof(unsigned int));
			__interal_spatialMap__::buildGrid<T, N_THREAD, 90> << <idxData / (32 * 6 * N_THREAD) + 1, dim3(32, 6) >> >(deviceData, deviceData2, idxData, deviceList, deviceGrid, m_deviceIdxCounter);
			ThrowCudaErrors(cudaDeviceSynchronize());
			swapk(deviceData, deviceData2);
		}
#endif
	}

	unsigned int getNumEntries() const
	{
		return numData;
	}

	unsigned int getNumStoredEntries() const
	{
		return idxData;
	}

	CUDA_FUNC_IN bool isFull() const
	{
		return idxData >= numData;
	}

#ifdef __CUDACC__
	CUDA_ONLY_FUNC bool store(const Vec3u& p, const T& v)
	{
		unsigned int data_idx = atomicInc(&idxData, (unsigned int)-1);
		if (data_idx >= numData)
			return false;
		unsigned int map_idx = hashMap.Hash(p);
		deviceData[data_idx] = v;
		deviceList[data_idx] = Vec2u(data_idx, map_idx);
		return true;
	}
#endif

	CUDA_ONLY_FUNC bool store(const Vec3f& p, const T& v)
	{
		return store(hashMap.Transform(p), v);
	}

	template<unsigned int MAX_ENTRIES_PER_CELL = UINT_MAX, typename CLB> CUDA_FUNC_IN void ForAllCellEntries(const Vec3u& p, const CLB& clb)
	{
		unsigned int map_idx = deviceGrid[hashMap.Hash(p)], i = 0;
		while (map_idx < idxData && i++ < MAX_ENTRIES_PER_CELL)
		{
			T& val = operator()(map_idx);
			clb(map_idx, val);
			map_idx = val.getFlag() ? UINT_MAX : map_idx + 1;
		}
	}

	CUDA_FUNC_IN const T& operator()(unsigned int idx) const
	{
		return deviceData[idx];
	}

	CUDA_FUNC_IN T& operator()(unsigned int idx)
	{
		return deviceData[idx];
	}
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
