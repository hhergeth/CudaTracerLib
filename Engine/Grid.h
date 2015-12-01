#pragma once

#include <Math/AABB.h>

namespace CudaTracerLib {

template<bool REGULAR> struct HashGrid
{
	unsigned int m_fGridSize;
	Vec3f m_vMin;
	Vec3f m_vInvSize;
	Vec3f m_vCellSize;
	AABB m_sBox;
	unsigned int N;

	CUDA_FUNC_IN HashGrid(){}

	CUDA_FUNC_IN HashGrid(const AABB& box, float a_InitialRadius, unsigned int a_NumEntries)
		: N(a_NumEntries - 1)
	{
		Vec3f q = (box.maxV - box.minV) / 2.0f, m = (box.maxV + box.minV) / 2.0f;
		float e = 0.015f, e2 = 1.0f + e;
		m_sBox.maxV = m + q * e2;
		m_sBox.minV = m - q * e2;
		m_fGridSize = (int)math::floor(math::pow(a_NumEntries, 1.0f / 3.0f));
		m_vMin = m_sBox.minV;
		m_vInvSize = Vec3f(1.0f) / m_sBox.Size() * (m_fGridSize - 1);
		m_vCellSize = m_sBox.Size() / (m_fGridSize - 1);
	}

	CUDA_FUNC_IN unsigned int hash6432shift(unsigned long long key) const
	{
		key = (~key) + (key << 18); // key = (key << 18) - key - 1;
		key = key ^ (key >> 31);
		key = key * 21; // key = (key + (key << 2)) + (key << 4);
		key = key ^ (key >> 11);
		key = key + (key << 6);
		return (unsigned int)(key ^ (key >> 2));
	}

	CUDA_FUNC_IN unsigned int Hash(const Vec3u& p) const
	{
		if (REGULAR)
		{
			unsigned int h = p.z * m_fGridSize * m_fGridSize + p.y * m_fGridSize + p.x;
			if (h > N)
				printf("h = %d, p = {%d, %d, %d}\n", h, p.x, p.y, p.z);
			return h;
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

			return (unsigned int)math::floor(math::frac(dot(n / m, Vec4f(1.0, -1.0, 1.0, -1.0))) * N);
		}
	}

	CUDA_FUNC_IN Vec3u InvHash(unsigned int idx) const
	{
		if (REGULAR)
		{
			unsigned int k = idx / (m_fGridSize * m_fGridSize);
			unsigned int j = (idx - k * m_fGridSize * m_fGridSize) / m_fGridSize;
			return Vec3u(idx - k * m_fGridSize * m_fGridSize - j * m_fGridSize, j, k);
		}
		else
		{
			return Vec3u(UINT_MAX);
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
		return clamp(Vec3u((unsigned int)q.x, (unsigned int)q.y, (unsigned int)q.z), Vec3u(0), Vec3u(m_fGridSize - 1));
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
		return (int(m.x) << 16) | (int(m.y) << 8) | int(m.z);
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

	CUDA_FUNC_IN AABB getCell(const Vec3u& cell_idx) const
	{
		Vec3f f = m_sBox.minV + m_vCellSize * Vec3f(cell_idx.x, cell_idx.y, cell_idx.z);
		return AABB(f, f + m_vCellSize);
	}
};

typedef HashGrid<true> HashGrid_Reg;
typedef HashGrid<false> HashGrid_Irreg;

template<typename V, typename T> CUDA_FUNC_IN V sign(T f)
{
	return f > T(0) ? V(1) : (f < T(0) ? V(-1) : V(0));
}
template<typename F> CUDA_FUNC_IN void TraverseGrid(const Ray& r, float tmin, float tmax, F& clb,
	const AABB& box, const Vec3f& gridSize)
{
	/*
	pbrt grid accellerator copy! (slightly streamlined for SIMD)
	*/
	Vec3f m_vCellSize = box.Size() / (gridSize - 1);
	float rayT, maxT;
	if (!box.Intersect(r, &rayT, &maxT))
		return;
	float minT = rayT = math::clamp(rayT, tmin, tmax);
	maxT = math::clamp(maxT, tmin, tmax);
	Vec3f q = (r(rayT) - box.minV) / box.Size() * (gridSize - Vec3f(1));
	Vec3u Pos = clamp(Vec3u((unsigned int)q.x, (unsigned int)q.y, (unsigned int)q.z), Vec3u(0), Vec3u(gridSize.x - 1, gridSize.y - 1, gridSize.z - 1));
	Vec3i Step(sign<int>(r.direction.x), sign<int>(r.direction.y), sign<int>(r.direction.z));
	Vec3f inv_d = r.direction;
	const float ooeps = math::exp2(-20.0f);//80 is too small, will create underflow on GPU
	inv_d.x = 1.0f / (math::abs(inv_d.x) > ooeps ? inv_d.x : copysignf(ooeps, inv_d.x));
	inv_d.y = 1.0f / (math::abs(inv_d.y) > ooeps ? inv_d.y : copysignf(ooeps, inv_d.y));
	inv_d.z = 1.0f / (math::abs(inv_d.z) > ooeps ? inv_d.z : copysignf(ooeps, inv_d.z));
	Vec3f NextCrossingT = Vec3f(rayT) + (box.minV + (Vec3f(Pos.x, Pos.y, Pos.z) + max(Vec3f(0.0f), sign(r.direction))) * m_vCellSize - r(rayT)) * inv_d,
		DeltaT = abs(m_vCellSize * inv_d);
	bool cancelTraversal = false;
	for (; !cancelTraversal;)
	{
		int bits = ((NextCrossingT[0] < NextCrossingT[1]) << 2) + ((NextCrossingT[0] < NextCrossingT[2]) << 1) + ((NextCrossingT[1] < NextCrossingT[2]));
		const int cmpToAxis[8] = { 2, 1, 2, 1, 2, 2, 0, 0 };
		int stepAxis = cmpToAxis[bits];
		clb(minT, rayT, maxT, min(NextCrossingT[stepAxis], maxT), Pos, cancelTraversal);
		Pos[stepAxis] += Step[stepAxis];
		if (Pos[stepAxis] >= gridSize[stepAxis] || maxT < NextCrossingT[stepAxis])
			break;
		rayT = NextCrossingT[stepAxis];
		NextCrossingT[stepAxis] += DeltaT[stepAxis];
	}
}

template<typename F> CUDA_FUNC_IN void TraverseGrid(const Ray& r, const HashGrid_Reg& grid, float tmin, float tmax, F& clb)
{
	return TraverseGrid(r, tmin, tmax, clb, grid.m_sBox, Vec3f(grid.m_fGridSize));
}

}
