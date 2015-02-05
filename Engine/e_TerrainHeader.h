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
	Vec3f m_sMin;
	Vec3f m_sMax;
	unsigned int m_uDepth;
	e_TerrainData_Inner* m_pNodes;
	CACHE_LEVEL_TYPE* m_pCacheData;
	CUDA_FUNC_IN Vec2f getsdxy()
	{
		Vec3f dim = m_sMax - m_sMin;
		return Vec2f(dim.x, dim.z) / Vec2f(pow2((float)m_uDepth));
	}
	Vec2f getFlatScale()
	{
		return Vec2f(m_sMax.x - m_sMin.x, m_sMax.z - m_sMin.z);
	}
};