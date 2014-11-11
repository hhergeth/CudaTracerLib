#pragma once

#include "e_Grid.h"

template<typename T> struct e_SpatialLinkedMap_volume_iterator;

//a mapping from R^3 -> T^n, ie. associating variable number of values with each point in the grid
template<typename T> struct e_SpatialLinkedMap
{
	struct linkedEntry
	{
		unsigned int nextIdx;
		T value;
	};

	unsigned int numData, gridSize;
	linkedEntry* deviceData;
	unsigned int* deviceMap;
	unsigned int deviceDataIdx;
	k_HashGrid_Reg hashMap;
public:
	typedef e_SpatialLinkedMap_volume_iterator<T> iterator;

	e_SpatialLinkedMap(){}
	e_SpatialLinkedMap(unsigned int gridSize, unsigned int numData)
		: numData(numData), gridSize(gridSize)
	{
		CUDA_MALLOC(&deviceData, sizeof(linkedEntry) * numData);
		CUDA_MALLOC(&deviceMap, sizeof(unsigned int) * gridSize * gridSize * gridSize);
	}

	void SetSceneDimensions(const AABB& box, float initialRadius)
	{
		hashMap = k_HashGrid_Reg(box, initialRadius, gridSize * gridSize * gridSize);
	}

	void ResetBuffer()
	{
		deviceDataIdx = 0;
		cudaMemset(deviceMap, -1, sizeof(unsigned int) * gridSize * gridSize * gridSize);
	}

	CUDA_FUNC_IN void store(const float3& p, const T& v)
	{
		//build linked list and spatial map
		unsigned int data_idx = Platform::Increment(&deviceDataIdx);
		if(deviceDataIdx >= numData)
			return;
		unsigned int map_idx = hashMap.Hash(p);
		unsigned int old_idx = Platform::Exchange(deviceMap + map_idx, data_idx);
		//copy actual data
		deviceData[data_idx].value = v;
		deviceData[data_idx].nextIdx = old_idx;
	}

	CUDA_FUNC_IN e_SpatialLinkedMap_volume_iterator<T> begin(const float3& p) const;

	CUDA_FUNC_IN e_SpatialLinkedMap_volume_iterator<T> end(const float3& max) const;

	CUDA_FUNC_IN e_SpatialLinkedMap_volume_iterator<T> begin(const float3& min, const float3& max) const;

	CUDA_FUNC_IN e_SpatialLinkedMap_volume_iterator<T> end(const float3& min, const float3& max) const;

	//internal

	CUDA_FUNC_IN unsigned int idx(const uint3& i) const
	{
		return deviceMap[hashMap.Hash(i)];
	}

	CUDA_FUNC_IN unsigned int nextIdx(unsigned int idx) const
	{
		return deviceData[idx].nextIdx;
	}

	CUDA_FUNC_IN const T& operator()(unsigned int idx) const
	{
		return deviceData[idx].value;
	}

	CUDA_FUNC_IN T& operator()(unsigned int idx)
	{
		return deviceData[idx];
	}
};

template<typename T> struct e_SpatialLinkedMap_volume_iterator
{
	const e_SpatialLinkedMap<T>& map;
	uint3 low, high, diff;// := [low, high)
	unsigned int dataIdx, flatGridIdx;

	CUDA_FUNC_IN e_SpatialLinkedMap_volume_iterator(const e_SpatialLinkedMap<T>& m, const uint3& mi, const uint3& ma, bool isEnd)
		: map(m), low(mi), high(ma + make_uint3(1)), diff(high - low)
	{
		flatGridIdx = isEnd ? diff.x * diff.y * diff.z : 0;
		dataIdx = isEnd ? unsigned int(-1) : m.idx(mi);
		if(!isEnd && dataIdx == unsigned int(-1))
			operator++();
	}
	
	CUDA_FUNC_IN e_SpatialLinkedMap_volume_iterator<T>& operator++()
	{
		if(dataIdx != unsigned int(-1))
			dataIdx = map.nextIdx(dataIdx);
		while(dataIdx == unsigned int(-1) && ++flatGridIdx != diff.x * diff.y * diff.z)
		{
			unsigned int slice = diff.x * diff.y, inSlice = flatGridIdx % slice;
			uint3 mi = low + make_uint3(inSlice % diff.x, inSlice / diff.x, flatGridIdx / slice);
			dataIdx = map.idx(mi);
		}
		return *this;
	}

	CUDA_FUNC_IN const T& operator*() const
	{
		return map(dataIdx);
	}

	CUDA_FUNC_IN const T* operator->() const
	{
		return &map(dataIdx);
	}

	CUDA_FUNC_IN bool operator==(const e_SpatialLinkedMap_volume_iterator<T>& rhs) const
	{
		return dataIdx == rhs.dataIdx && flatGridIdx == rhs.flatGridIdx;
	}

	CUDA_FUNC_IN bool operator!=(const e_SpatialLinkedMap_volume_iterator<T>& rhs) const
	{
		return !operator==(rhs);
	}
};

template<typename T> e_SpatialLinkedMap_volume_iterator<T> e_SpatialLinkedMap<T>::begin(const float3& p) const
{
	return e_SpatialLinkedMap_volume_iterator<T>(*this, hashMap.Transform(p), hashMap.Transform(p), false);
}

template<typename T> e_SpatialLinkedMap_volume_iterator<T> e_SpatialLinkedMap<T>::end(const float3& p) const
{
	return e_SpatialLinkedMap_volume_iterator<T>(*this, hashMap.Transform(p), hashMap.Transform(p), true);
}

template<typename T> e_SpatialLinkedMap_volume_iterator<T> e_SpatialLinkedMap<T>::begin(const float3& min, const float3& max) const
{
	return e_SpatialLinkedMap_volume_iterator<T>(*this, hashMap.Transform(min), hashMap.Transform(max), false);
}

template<typename T> e_SpatialLinkedMap_volume_iterator<T> e_SpatialLinkedMap<T>::end(const float3& min, const float3& max) const
{
	return e_SpatialLinkedMap_volume_iterator<T>(*this, hashMap.Transform(min), hashMap.Transform(max), true);
}

//a mapping from R^3 -> T, ie. associating one element with each point in the grid
template<typename T> struct e_SpatialSet
{
	unsigned int gridSize;
	T* deviceData;
	k_HashGrid_Reg hashMap;
public:
	e_SpatialSet(){}
	e_SpatialSet(unsigned int gridSize)
		: gridSize(gridSize)
	{
		CUDA_MALLOC(&deviceData, sizeof(T) * gridSize * gridSize * gridSize);
	}

	void SetSceneDimensions(const AABB& box, float initialRadius)
	{
		hashMap = k_HashGrid_Reg(box, initialRadius, gridSize * gridSize * gridSize);
	}

	void ResetBuffer()
	{
		cudaMemset(deviceData, 0, sizeof(T) * gridSize * gridSize * gridSize);
	}

	CUDA_FUNC_IN const T& operator()(const float3& p) const
	{
		return deviceData[hashMap.Hash(p)].value;
	}

	CUDA_FUNC_IN T& operator()(const float3& p)
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