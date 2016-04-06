#pragma once

#include "Grid.h"
#include "SynchronizedBuffer.h"
#include <Base/Timer.h>
#ifdef __CUDACC__
#pragma warning (disable : 4267) 
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#pragma warning (default : 4267)
#endif

namespace CudaTracerLib {

template<typename T, typename HASHER> class SpatialGrid
{
protected:
	HashGrid_Reg hashMap;
public:
	virtual ~SpatialGrid()
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
		Vec3u a = min(min_cell, hashMap.m_gridDim - Vec3u(1)), b = min(max_cell, hashMap.m_gridDim - Vec3u(1));
		for (unsigned int ax = a.x; ax <= b.x; ax++)
			for (unsigned int ay = a.y; ay <= b.y; ay++)
				for (unsigned int az = a.z; az <= b.z; az++)
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
		ForAllCells(Vec3u(0), hashMap.m_gridDim - Vec3u(1), clb);
	}
};

//a mapping from R^3 -> T^n, ie. associating variable number of values with each point in the grid
template<typename T> class SpatialLinkedMap : public SpatialGrid<T, SpatialLinkedMap<T>>, public ISynchronizedBufferParent
{
typedef SpatialGrid<T, SpatialLinkedMap<T>> BaseType;
public:
	struct linkedEntry
	{
		unsigned int nextIdx;
		T value;
	};
private:
	unsigned int numData;
	Vec3u m_gridSize;
	unsigned int deviceDataIdx;
	SynchronizedBuffer<linkedEntry> m_dataBuffer;
	SynchronizedBuffer<unsigned int> m_mapBuffer;
public:
	SpatialLinkedMap(const Vec3u& gridSize, unsigned int numData)
		: ISynchronizedBufferParent(m_dataBuffer, m_mapBuffer), numData(numData), m_gridSize(gridSize),
		  m_dataBuffer(numData), m_mapBuffer(m_gridSize.x * m_gridSize.y * m_gridSize.z)
	{
		m_dataBuffer.Memset(0xff);
	}

	void SetSceneDimensions(const AABB& box)
	{
		BaseType::hashMap = HashGrid_Reg(box, m_gridSize);
	}

	void ResetBuffer()
	{
		deviceDataIdx = 0;
		m_mapBuffer.Memset((unsigned char)0xff);
	}

	CUDA_FUNC_IN unsigned int getNumEntries() const
	{
		return numData;
	}

	CUDA_FUNC_IN unsigned int getNumStoredEntries() const
	{
		return deviceDataIdx;
	}

	void PrepareForUse() {}

	CUDA_FUNC_IN bool isFull() const
	{
		return deviceDataIdx >= numData;
	}

	CUDA_FUNC_IN void store(const Vec3u& p, const T& v, unsigned int data_idx)
	{
		unsigned int map_idx = BaseType::hashMap.Hash(p);
#ifdef ISCUDA
		unsigned int old_idx = atomicExch(&m_mapBuffer[map_idx], data_idx);
#else
		unsigned int old_idx = Platform::Exchange(&m_mapBuffer[map_idx], data_idx);
#endif
		//copy actual data
		m_dataBuffer[data_idx].value = v;
		m_dataBuffer[data_idx].nextIdx = old_idx;
	}

	CUDA_FUNC_IN bool store(const Vec3u& p, const T& v)
	{
		//build linked list and spatial map
#ifdef ISCUDA
		unsigned int data_idx = atomicInc(&deviceDataIdx, (unsigned int)-1);
#else
		unsigned int data_idx = Platform::Increment(&deviceDataIdx);
#endif	
		if (data_idx >= numData)
			return false;
		unsigned int map_idx = BaseType::hashMap.Hash(p);
#ifdef ISCUDA
		unsigned int old_idx = atomicExch(&m_mapBuffer[map_idx], data_idx);
#else
		unsigned int old_idx = Platform::Exchange(&m_mapBuffer[map_idx], data_idx);
#endif
		//copy actual data
		m_dataBuffer[data_idx].value = v;
		m_dataBuffer[data_idx].nextIdx = old_idx;
		return true;
	}

	CUDA_FUNC_IN bool store(const Vec3f& p, const T& v)
	{
		return store(BaseType::hashMap.Transform(p), v);
	}

	CUDA_FUNC_IN unsigned int allocStorage(unsigned int n)
	{
#ifdef ISCUDA
		unsigned int idx = atomicAdd(&deviceDataIdx, n);
#else
		unsigned int idx = Platform::Add(&deviceDataIdx, n);
#endif
		return idx;
	}

	template<unsigned int MAX_ENTRIES_PER_CELL = UINT_MAX, typename CLB> CUDA_FUNC_IN void ForAllCellEntries(const Vec3u& p, const CLB& clb)
	{
		unsigned int i0 = BaseType::hashMap.Hash(p), i = m_mapBuffer[i0], N = 0, lo = min(deviceDataIdx, numData);
		while (i < lo && N++ < MAX_ENTRIES_PER_CELL)
		{
			clb(i, m_dataBuffer[i].value);
			i = m_dataBuffer[i].nextIdx;
		}
	}

	CUDA_FUNC_IN const T& operator()(unsigned int idx) const
	{
		return m_dataBuffer[idx].value;
	}

	CUDA_FUNC_IN T& operator()(unsigned int idx)
	{
		return m_dataBuffer[idx].value;
	}

	linkedEntry* getDeviceData() { return m_dataBuffer.getDevicePtr(); }
	unsigned int* getDeviceGrid() { return m_mapBuffer.getDevicePtr(); }
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
		deviceDataDest[tarBufferLoc - 1].setFlag();
	}
}

}
#endif

template<typename T> class SpatialFlatMap : public SpatialGrid<T, SpatialFlatMap<T>>, public ISynchronizedBufferParent
{
typedef SpatialGrid<T, SpatialFlatMap<T>> BaseType;
public:
	Vec3u m_gridSize;
	unsigned int numData, idxData;
	unsigned int* m_deviceIdxCounter;
	SynchronizedBuffer<T> m_buffer1, m_buffer2;
	SynchronizedBuffer<unsigned int> m_gridBuffer;
	SynchronizedBuffer<Vec2u> m_listBuffer;

	SpatialFlatMap(const Vec3u& gridSize, unsigned int numData)
		: ISynchronizedBufferParent(m_buffer1, m_buffer2, m_gridBuffer, m_listBuffer),
		  numData(numData), m_gridSize(gridSize), idxData(0),
		  m_buffer1(numData), m_buffer2(numData), m_gridBuffer(m_gridSize.x * m_gridSize.y * m_gridSize.z), m_listBuffer(numData)
	{
		CUDA_MALLOC(&m_deviceIdxCounter, sizeof(unsigned int));
	}

	virtual void Free() override
	{
		CUDA_FREE(m_deviceIdxCounter);
		ISynchronizedBufferParent::Free();
	}

	void SetSceneDimensions(const AABB& box)
	{
		BaseType::hashMap = HashGrid_Reg(box, m_gridSize);
	}

	void ResetBuffer()
	{
		idxData = 0;
	}

	void PrepareForUse()
	{
		auto& Tt = GET_PERF_BLOCKS();

		idxData = min(idxData, numData);
		auto GP = m_gridSize.x * m_gridSize.y * m_gridSize.z;
		
#ifndef __CUDACC__
		throw std::runtime_error("Use this from a cuda file please!"); 
		/*
		{
			auto bl = Tt.StartBlock("sort");
			thrust::sort(thrust::device_ptr<Vec2u>(m_listBuffer.getDevicePtr()), thrust::device_ptr<Vec2u>(m_listBuffer.getDevicePtr() + idxData), __interal_spatialMap__::order());
			m_listBuffer.Synchronize();
		}
		{
			auto bl = Tt.StartBlock("reset");
			m_gridBuffer.Memset((unsigned char)0xff);
		}
		{
			auto bl = Tt.StartBlock("build");
			m_buffer1.Synchronize();
			unsigned int i = 0;
			while (i < idxData)
			{
				unsigned int cellHash = m_listBuffer[i].y;
				m_gridBuffer[cellHash] = i;
				while (i < idxData && m_listBuffer[i].y == cellHash)
				{
					m_buffer2[i] = m_buffer1[m_listBuffer[i].x];
					i++;
				}
				m_buffer2[i - 1].setFlag();
			}
			m_gridBuffer.setOnCPU(); m_gridBuffer.Synchronize();
			m_buffer2.setOnCPU(); m_buffer2.Synchronize();
		}
		*/
#else
		{
			auto bl = Tt.StartBlock("sort");
			thrust::sort(thrust::device_ptr<Vec2u>(m_listBuffer.getDevicePtr()), thrust::device_ptr<Vec2u>(m_listBuffer.getDevicePtr() + idxData), __interal_spatialMap__::order());
		}
		{
			auto bl = Tt.StartBlock("reset");
			ThrowCudaErrors(cudaMemset(deviceGrid, 0xffffffff, GP));
		}
		{
			auto bl = Tt.StartBlock("build");
			const unsigned int N_THREAD = 10;
			CudaSetToZero(m_deviceIdxCounter, sizeof(unsigned int));
			__interal_spatialMap__::buildGrid<T, N_THREAD, 90> << <idxData / (32 * 6 * N_THREAD) + 1, dim3(32, 6) >> >
				(m_buffer1.getDevicePtr(), m_buffer2.getDevicePtr(), idxData, m_listBuffer.getDevicePtr(), m_gridBuffer.getDevicePtr(), m_deviceIdxCounter);
			ThrowCudaErrors(cudaDeviceSynchronize());
		}
#endif
		swapk(m_buffer1, m_buffer2);
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
		unsigned int map_idx = BaseType::hashMap.Hash(p);
		m_buffer1[data_idx] = v;
		m_listBuffer[data_idx] = Vec2u(data_idx, map_idx);
		return true;
	}
#endif

	CUDA_ONLY_FUNC bool store(const Vec3f& p, const T& v)
	{
		return store(BaseType::hashMap.Transform(p), v);
	}

	template<unsigned int MAX_ENTRIES_PER_CELL = UINT_MAX, typename CLB> CUDA_FUNC_IN void ForAllCellEntries(const Vec3u& p, const CLB& clb)
	{
		unsigned int map_idx = m_gridBuffer[BaseType::hashMap.Hash(p)], i = 0;
		while (map_idx < idxData && i++ < MAX_ENTRIES_PER_CELL)
		{
			T& val = operator()(map_idx);
			clb(map_idx, val);
			map_idx = val.getFlag() ? UINT_MAX : map_idx + 1;
		}
	}

	CUDA_FUNC_IN const T& operator()(unsigned int idx) const
	{
		return m_buffer1[idx];
	}

	CUDA_FUNC_IN T& operator()(unsigned int idx)
	{
		return m_buffer1[idx];
	}
};

//a mapping from R^3 -> T, ie. associating one element with each point in the grid
template<typename T> struct SpatialSet : public ISynchronizedBufferParent
{
	Vec3u m_gridSize;
	HashGrid_Reg hashMap;
	SynchronizedBuffer<T> m_buffer;
public:
	SpatialSet(const Vec3u& gridSize)
		: ISynchronizedBufferParent(m_buffer), m_gridSize(gridSize), m_buffer(m_gridSize.x * m_gridSize.y * m_gridSize.z)
	{
	}

	void SetSceneDimensions(const AABB& box)
	{
		hashMap = HashGrid_Reg(box, m_gridSize);
	}

	void ResetBuffer()
	{
		m_buffer.Memset((unsigned char)0);
	}

	CUDA_FUNC_IN const T& operator()(const Vec3f& p) const
	{
		return m_buffer[hashMap.Hash(p)].value;
	}

	CUDA_FUNC_IN T& operator()(const Vec3f& p)
	{
		return m_buffer[hashMap.Hash(p)];
	}

	CUDA_FUNC_IN const T& operator()(unsigned int idx) const
	{
		return m_buffer[idx];
	}

	CUDA_FUNC_IN T& operator()(unsigned int idx)
	{
		return m_buffer[idx];
	}

	CUDA_FUNC_IN unsigned int NumEntries() const
	{
		return m_gridSize.x * m_gridSize.y * m_gridSize.z;
	}
};

}
