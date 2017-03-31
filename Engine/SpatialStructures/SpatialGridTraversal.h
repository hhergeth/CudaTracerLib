#pragma once

#include <Math/Ray.h>
#include <Math/AABB.h>
#include <Engine/SpatialStructures/SpatialGrid.h>

namespace CudaTracerLib {

template<typename F> CUDA_FUNC_IN void TraverseGridRay(const Ray& r, float tmin, float tmax, const AABB& box, const Vec3f& gridSize, F clb)
{
	/*
	pbrt grid accellerator copy! (slightly streamlined for SIMD)
	*/

	auto _sign = [](float f) { return f > 0.0f ? 1 : (f < 0.0f ? -1 : 0); };

	Vec3f m_vCellSize = box.Size() / gridSize;
	float rayT, maxT;
	if (!box.Intersect(r, &rayT, &maxT))
		return;
	float minT = rayT = math::clamp(rayT, tmin, tmax);
	maxT = math::clamp(maxT, tmin, tmax);
	Vec3f q = (r(rayT) - box.minV) / m_vCellSize;
	Vec3u Pos = (clamp(q, Vec3f(0.0f), gridSize - Vec3f(1))).floor_u();
	Vec3i Step(_sign(r.dir().x), _sign(r.dir().y), _sign(r.dir().z));
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

template<typename F> CUDA_FUNC_IN void TraverseGridRay(const Ray& r, float tmin, float tmax, const HashGrid_Reg& grid, F clb)
{
	return TraverseGridRay(r, tmin, tmax, grid.m_sBox, Vec3f(grid.m_gridDim), clb);
}

template<template<class> class Grid, typename T, typename F1, typename F2, typename F3, typename F4> CUDA_FUNC_IN void TraverseGridBeamExt(const Ray& r, float tmin, float tmax, Grid<T>& grid, F1 clbRad, F2 clbDist,
																																		   F3 clbElement, F4 clbEndCell)
{
	auto last_min = Vec3u(0xffffffff), last_max = Vec3u(0xffffffff);
	TraverseGridRay(r, tmin, tmax, grid.getHashGrid().getAABB(), Vec3f(grid.getHashGrid().m_gridDim), [&](float minT, float rayT, float maxT, float cellEndT, const Vec3u& cell_pos, bool& cancelTraversal)
	{
		auto rad = clbRad(cell_pos, rayT, cellEndT);
		auto a = r(rayT), b = r(cellEndT);
		auto search_box = AABB(min(a, b) - Vec3f(rad), max(a, b) + Vec3f(rad));
		auto idx_min_cell = grid.getHashGrid().Transform(search_box.minV);
		auto idx_max_cell = grid.getHashGrid().Transform(search_box.maxV);

		grid.ForAllCells(idx_min_cell, idx_max_cell, [&](const Vec3u& cell_idx)
		{
			//check if cell was already visited
			//if (last_min.x <= cell_idx.x && cell_idx.x <= last_max.x &&
			//	last_min.y <= cell_idx.y && cell_idx.y <= last_max.y &&
			//	last_min.z <= cell_idx.z && cell_idx.z <= last_max.z)
			//	return;

			grid.ForAllCellEntries(cell_idx, [&](unsigned int element_idx, const T& element)
			{
				float distAlongRay = -1;
				auto dist_data = clbDist(cell_idx, element_idx, element, distAlongRay, rad);
				if ((rayT <= distAlongRay || rayT <= minT * (1 + EPSILON)) && distAlongRay >= tmin  && distAlongRay <= tmax && (distAlongRay < cellEndT || cellEndT >= maxT * (1.0f - EPSILON)))
				{
					clbElement(rayT, cellEndT, minT, maxT, cell_idx, element_idx, element, distAlongRay, dist_data, rad);
				}
			});
		});

		clbEndCell(rayT, cellEndT, minT, maxT, cell_pos);

		last_min = idx_min_cell;
		last_max = idx_max_cell;
	});
}
template<template<class> class Grid, typename T, typename F1, typename F2, typename F3> CUDA_FUNC_IN void TraverseGridBeam(const Ray& r, float tmin, float tmax, Grid<T>& grid, F1 clbRad, F2 clbDist, F3 clbElement)
{
	return TraverseGridBeamExt(r, tmin, tmax, grid, clbRad, clbDist, clbElement, [&](float rayT, float cellEndT, float minT, float maxT, const Vec3u& cell_idx) { });
}


CUDA_FUNC_IN float sqrDistanceToRay(const Ray& r, const Vec3f& pos, float& distanceAlongRay)
{
	distanceAlongRay = dot(pos - r.ori(), r.dir());
	return distanceSquared(pos, r(distanceAlongRay));
}

CUDA_FUNC_IN bool sphere_line_intersection(const Vec3f& p, float radSqr, const Ray& r, float& t_min, float& t_max)
{
	auto d = r.dir(), o = r.ori();
	float a = lenSqr(d), b = 2 * dot(d, o - p), c = lenSqr(p) + lenSqr(o) - 2 * dot(p, o) - radSqr;
	float disc = b * b - 4 * a* c;
	if (disc < 0)
		return false;
	float q = math::sqrt(disc);
	t_min = (-b - q) / (2 * a);
	t_max = (-b + q) / (2 * a);
	return true;
}

}