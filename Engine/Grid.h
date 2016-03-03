#pragma once

#include <Math/AABB.h>

namespace CudaTracerLib {

//leaves the first \arg LEFT_BITS bits in the result empty
template<int LEFT_BITS = 0, typename STORAGE_TYPE> CUDA_FUNC_IN STORAGE_TYPE EncodePos(const AABB& box, const Vec3f& pos)
{
	const unsigned int n_bits_per_component = (sizeof(STORAGE_TYPE) * 8 - LEFT_BITS) / 3;
	const float scale = (float)((1 << n_bits_per_component) - 1);
	const auto rel_pos = (pos - box.minV) / box.Size();
	STORAGE_TYPE x = (STORAGE_TYPE)(rel_pos.x * scale), y = (STORAGE_TYPE)(rel_pos.y * scale), z = (STORAGE_TYPE)(rel_pos.z * scale);
	return ((x << (2 * n_bits_per_component)) | (y << n_bits_per_component) | z) << LEFT_BITS;
}

template<int LEFT_BITS = 0, typename STORAGE_TYPE> CUDA_FUNC_IN Vec3f DecodePos(const AABB& box, STORAGE_TYPE pos_encoded)
{
	const unsigned int n_bits_per_component = (sizeof(STORAGE_TYPE) * 8 - LEFT_BITS) / 3;
	const STORAGE_TYPE mask = (1 << n_bits_per_component) - 1;
	pos_encoded = pos_encoded >> LEFT_BITS;
	STORAGE_TYPE x = (pos_encoded >> (2 * n_bits_per_component)) & mask, y = (pos_encoded >> n_bits_per_component) & mask, z = pos_encoded & mask;
	return math::lerp(box.minV, box.maxV, Vec3f((float)x, (float)y, (float)z) / mask);
}

template<bool REGULAR> struct HashGrid
{
	Vec3u m_gridDim;
	Vec3f m_vInvSize;
	AABB m_sBox;
	unsigned int m_nElements;
	Vec3f m_vCellSize;

	CUDA_FUNC_IN HashGrid(){}

	CUDA_FUNC_IN HashGrid(const AABB& box, const Vec3u& gridSize)
		: m_sBox(box), m_nElements(gridSize.x * gridSize.y * gridSize.z), m_gridDim(gridSize)
	{
#ifndef ISCUDA
		if (gridSize.max() > 1024)
			throw std::runtime_error("Cannot create grids of size > 1024 due to issues with flattening index!");
#endif
		m_vCellSize = m_sBox.Size() / Vec3f((float)m_gridDim.x, (float)m_gridDim.y, (float)m_gridDim.z);
		m_vInvSize = Vec3f(1.0f) / m_vCellSize;
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
		if (p.x >= m_gridDim.x || p.y >= m_gridDim.y || p.z >= m_gridDim.z)
			return UINT_MAX;

		if (REGULAR)
		{
			return p.z * m_gridDim.x * m_gridDim.y + p.y * m_gridDim.x + p.x;
		}
		else
		{
			// use the same procedure as GPURnd
			Vec4f n = Vec4f((float)p.x, (float)p.y, (float)p.z, (float)(p.x + p.y - p.z)) *  4294967295.0f;

			const Vec4f q = Vec4f(1225.0, 1585.0, 2457.0, 2098.0);
			const Vec4f r = Vec4f(1112.0, 367.0, 92.0, 265.0);
			const Vec4f a = Vec4f(3423.0, 2646.0, 1707.0, 1999.0);
			const Vec4f m = Vec4f(4194287.0, 4194277.0, 4194191.0, 4194167.0);

			Vec4f beta = floor(n / q);
			Vec4f p = a * (n - beta * q) - beta * r;
			beta = (sign(p) + Vec4f(1.0)) * Vec4f(0.5) * m;
			n = (p + beta);

			return (unsigned int)math::floor(math::frac(dot(n / m, Vec4f(1.0, -1.0, 1.0, -1.0))) * (m_nElements - 1));
		}
	}

	CUDA_FUNC_IN Vec3u InvHash(unsigned int idx) const
	{
		if (idx >= m_nElements)
			return Vec3u(UINT_MAX);
		if (REGULAR)
		{
			unsigned int k = idx / (m_gridDim.x * m_gridDim.y), koff = k * m_gridDim.x * m_gridDim.y;
			unsigned int j = (idx - koff) / m_gridDim.x;
			return Vec3u(idx - koff - j * m_gridDim.x, j, k);
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
		Vec3f q = (p - m_sBox.minV) * m_vInvSize;
		q = clamp(q, Vec3f(0.0f), Vec3f((float)m_gridDim.x, (float)m_gridDim.y, (float)m_gridDim.z) - Vec3f(1));
		return Vec3u((unsigned int)q.x, (unsigned int)q.y, (unsigned int)q.z);
		//return clamp(Vec3u((unsigned int)q.x, (unsigned int)q.y, (unsigned int)q.z), Vec3u(0), m_gridDim - Vec3u(1));
	}

	CUDA_FUNC_IN Vec3f InverseTransform(const Vec3u& i) const
	{
		return Vec3f(i.x, i.y, i.z) * m_vCellSize + m_sBox.minV;
	}

	CUDA_FUNC_IN bool IsValidHash(const Vec3f& p) const
	{
		return m_sBox.Contains(p);
	}

	CUDA_FUNC_IN unsigned int EncodePos(const Vec3f& p, const Vec3u& i) const
	{
		Vec3f low = Vec3f((float)i.x, (float)i.y, (float)i.z) * m_vCellSize + m_sBox.minV;
		Vec3f m = clamp01((p - low) * m_vInvSize) * 255.0f;
		return (int(m.x) << 16) | (int(m.y) << 8) | int(m.z);
	}

	CUDA_FUNC_IN Vec3f DecodePos(unsigned int p, const Vec3u& i) const
	{
		unsigned int q = 0x00ff0000, q2 = 0x0000ff00, q3 = 0x000000ff;
		Vec3f low = Vec3f((float)i.x, (float)i.y, (float)i.z) * m_vCellSize + m_sBox.minV;
		Vec3f m = (Vec3f((float)((p & q) >> 16), (float)((p & q2) >> 8), (float)(p & q3)) / 255.0f) * m_vCellSize + low;
		return m;
	}

	CUDA_FUNC_IN unsigned int FlattenIndex(const Vec3u& idx) const
	{
		return (idx.x << 20) | (idx.y << 10) | (idx.z);
	}

	CUDA_FUNC_IN Vec3u ExpandIndex(unsigned int flat_idx) const
	{
		return Vec3u((flat_idx >> 20) & 0x3ff, (flat_idx >> 10) & 0x3ff, flat_idx & 0x3ff);
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
template<typename F> CUDA_FUNC_IN void TraverseGrid(const Ray& r, float tmin, float tmax, const F& clb,
	const AABB& box, const Vec3f& gridSize)
{
	/*
	pbrt grid accellerator copy! (slightly streamlined for SIMD)
	*/
	Vec3f m_vCellSize = box.Size() / gridSize;
	float rayT, maxT;
	if (!box.Intersect(r, &rayT, &maxT))
		return;
	float minT = rayT = math::clamp(rayT, tmin, tmax);
	maxT = math::clamp(maxT, tmin, tmax);
	Vec3f q = (r(rayT) - box.minV) / m_vCellSize;
	Vec3u Pos = (clamp(q, Vec3f(0.0f), gridSize - Vec3f(1))).floor_u();
	Vec3i Step(sign<int>(r.dir().x), sign<int>(r.dir().y), sign<int>(r.dir().z));
	Vec3f inv_d = r.dir();
	const float ooeps = math::exp2(-40.0f);
	inv_d.x = 1.0f / (math::abs(r.dir().x) > ooeps ? r.dir().x : copysignf(ooeps, r.dir().x));
	inv_d.y = 1.0f / (math::abs(r.dir().y) > ooeps ? r.dir().y : copysignf(ooeps, r.dir().y));
	inv_d.z = 1.0f / (math::abs(r.dir().z) > ooeps ? r.dir().z : copysignf(ooeps, r.dir().z));
	Vec3f NextCrossingT = Vec3f(rayT) + (box.minV + (Vec3f(Pos) + max(Vec3f(0.0f), sign(r.dir()))) * m_vCellSize - r(rayT)) * inv_d,
		DeltaT = abs(m_vCellSize * inv_d);
	bool cancelTraversal = false;
	for (; !cancelTraversal;)
	{
		int bits = ((NextCrossingT[0] < NextCrossingT[1]) << 2) + ((NextCrossingT[0] < NextCrossingT[2]) << 1) + ((NextCrossingT[1] < NextCrossingT[2]));	
		int stepAxis = (0x00000a66 >> bits * 2) & 3;//cmpToAxis[bits]; //const int cmpToAxis[8] = { 2, 1, 2, 1, 2, 2, 0, 0 };
		clb(minT, rayT, maxT, min(NextCrossingT[stepAxis], maxT), Pos, cancelTraversal);
		Pos[stepAxis] += Step[stepAxis];
		if (Pos[stepAxis] >= gridSize[stepAxis] || maxT < NextCrossingT[stepAxis])
			break;
		rayT = NextCrossingT[stepAxis];
		NextCrossingT[stepAxis] += DeltaT[stepAxis];
	}
}

template<typename F> CUDA_FUNC_IN void TraverseGrid(const Ray& r, const HashGrid_Reg& grid, float tmin, float tmax, const F& clb)
{
	return TraverseGrid(r, tmin, tmax, clb, grid.m_sBox, Vec3f(grid.m_gridDim));
}

}
