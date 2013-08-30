#pragma once

#include <MathTypes.h>

CUDA_FUNC_IN unsigned int pow2(unsigned int n)
{
	return 1 << n;
}

CUDA_FUNC_IN unsigned int pow4(unsigned int n)
{
	return 1 << (2 * n);
}

CUDA_FUNC_IN unsigned int sum_pow4(unsigned int n)
{
	unsigned int r = (pow4(n + 1) - 1) / 3;
	return r;
}

struct e_TerrainData_Inner;

#define CACHE_LEVEL 4
typedef float2 CACHE_LEVEL_TYPE;

struct e_KernelTerrainData
{
	float3 m_sMin;
	float3 m_sMax;
	unsigned int m_uDepth;
	e_TerrainData_Inner* m_pNodes;
	CACHE_LEVEL_TYPE* m_pCacheData;
	CUDA_FUNC_IN float2 getsdxy()
	{
		float3 dim = m_sMax - m_sMin;
		return make_float2(dim.x, dim.z) / make_float2(pow2(m_uDepth));
	}
	float2 getFlatScale()
	{
		return make_float2(m_sMax.x - m_sMin.x, m_sMax.z - m_sMin.z);
	}
};