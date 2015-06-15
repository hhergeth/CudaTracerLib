#pragma once

#include "..\Math\AABB.h"

template<bool REGULAR> struct k_HashGrid
{
	unsigned int m_fGridSize;
	Vec3f m_vMin;
	Vec3f m_vInvSize;
	Vec3f m_vCellSize;
	AABB m_sBox;
	unsigned int N;

	CUDA_FUNC_IN k_HashGrid(){}

	CUDA_FUNC_IN k_HashGrid(const AABB& box, float a_InitialRadius, unsigned int a_NumEntries)
		: N(a_NumEntries - 1)
	{
		Vec3f q = (box.maxV - box.minV) / 2.0f, m = (box.maxV + box.minV) / 2.0f;
		float e = 0.015f, e2 = 1.0f + e;
		m_sBox.maxV = m + q * e2;
		m_sBox.minV = m - q * e2;
		m_fGridSize = (int)floor(math::pow(a_NumEntries, 1.0f/3.0f));
		m_vMin = m_sBox.minV;
		m_vInvSize = Vec3f(1.0f) / m_sBox.Size() * m_fGridSize;
		m_vCellSize = m_sBox.Size() / m_fGridSize;
	}

	CUDA_FUNC_IN unsigned int hash6432shift(unsigned long long key) const
	{
		key = (~key) + (key << 18); // key = (key << 18) - key - 1;
		key = key ^ (key >>  31);
		key = key * 21; // key = (key + (key << 2)) + (key << 4);
		key = key ^ (key >> 11);
		key = key + (key << 6);
		return unsigned int(key) ^ unsigned int(key >> 2);
	}

	CUDA_FUNC_IN unsigned int Hash(const uint3& p) const
	{
		if(REGULAR)
		{
			return math::clamp((unsigned int)(p.z * m_fGridSize * m_fGridSize + p.y * m_fGridSize + p.x), 0u, N);
		}
		else
		{
			// use the same procedure as GPURnd
			Vec4f n = Vec4f(p.x, p.y, p.z, p.x + p.y - p.z) *  4294967295.0f;

			const Vec4f q = Vec4f(1225.0, 1585.0, 2457.0, 2098.0);
			const Vec4f r = Vec4f(1112.0, 367.0, 92.0, 265.0);
			const Vec4f a = Vec4f(3423.0, 2646.0, 1707.0, 1999.0);
			const Vec4f m = Vec4f(4194287.0, 4194277.0, 4194191.0, 4194167.0);

			Vec4f beta = floor(n / q);
			Vec4f p = a * (n - beta * q) - beta * r;
			beta = (sign(p) + Vec4f(1.0)) * Vec4f(0.5) * m;
			n = (p + beta);

			return (unsigned int)math::floor( math::frac(dot(n / m, Vec4f(1.0, -1.0, 1.0, -1.0))) * N );
		}
	}

	//small helper function which does Hash(Transform(p))
	CUDA_FUNC_IN unsigned int Hash(const Vec3f& p) const
	{
		return Hash(Transform(p));
	}

	CUDA_FUNC_IN  Vec3u Transform(const Vec3f& p) const
	{
		Vec3f q = (p - m_vMin) * m_vInvSize;
		return clamp(Vec3u(unsigned int(q.x), unsigned int(q.y), unsigned int(q.z)), Vec3u(0), Vec3u(m_fGridSize));
	}

	CUDA_FUNC_IN Vec3f InverseTransform(const Vec3u& i) const
	{
		return Vec3f(i.x, i.y, i.z) * m_vCellSize + m_vMin;
	}

	CUDA_FUNC_IN bool IsValidHash(const Vec3f& p) const
	{
		return m_sBox.Contains(p);
	}

	CUDA_FUNC_IN unsigned int EncodePos(const Vec3f& p, const Vec3u& i) const
	{
		Vec3f low = Vec3f(i.x, i.y, i.z) / m_vInvSize + m_vMin;
		Vec3f m = clamp01((p - low) / m_vCellSize) * 255.0f;
		return (unsigned int(m.x) << 16) | (unsigned int(m.y) << 8) | (unsigned int(m.z));
	}
	
	CUDA_FUNC_IN Vec3f DecodePos(unsigned int p, const Vec3u& i) const
	{
		unsigned int q = 0x00ff0000, q2 = 0x0000ff00, q3 = 0x000000ff;
		Vec3f low = Vec3f(i.x, i.y, i.z) / m_vInvSize + m_vMin;
		Vec3f m = (Vec3f((p & q) >> 16, (p & q2) >> 8, (p & q3)) / 255.0f) * m_vCellSize + low;
		return m;
	}

	CUDA_FUNC_IN AABB getAABB() const
	{
		return m_sBox;
	}
};

typedef k_HashGrid<true> k_HashGrid_Reg;
typedef k_HashGrid<false> k_HashGrid_Irreg;