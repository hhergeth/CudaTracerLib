#pragma once

#include <Math/Ray.h>
#include <Math/AABB.h>
#include <Engine/SpatialGrid.h>

namespace CudaTracerLib {

template<template<class> class Grid, typename T, typename F1, typename F2, typename F3, typename F4> CUDA_FUNC_IN void TraverseGridBeamExt(const Ray& r, float tmin, float tmax, Grid<T>& grid, const F1& clbRad, const F2& clbDist, 
																																		const F3& clbElement, const F4& clbEndCell)
{
	auto last_min = Vec3u(0xffffffff), last_max = Vec3u(0xffffffff);
	TraverseGridRay(r, tmin, tmax, grid.getHashGrid().getAABB(), Vec3f(grid.getHashGrid().m_gridDim), [&](float minT, float rayT, float maxT, float cellEndT, const Vec3u& cell_pos, bool& cancelTraversal)
	{
		auto rad = clbRad(cell_pos, rayT, cellEndT);
		auto a = r(rayT), b = r(cellEndT);
		auto search_box = AABB(min(a, b) - Vec3f(rad), max(a, b) + Vec3f(rad));
		auto idx_min_cell = grid.getHashGrid().Transform(search_box.minV),
			idx_max_cell = grid.getHashGrid().Transform(search_box.maxV);

		grid.ForAllCells(idx_min_cell, idx_max_cell, [&](const Vec3u& cell_idx)
		{
			//check if cell was already visited
			if (last_min.x <= cell_idx.x && cell_idx.x <= last_max.x &&
				last_min.y <= cell_idx.y && cell_idx.y <= last_max.y &&
				last_min.z <= cell_idx.z && cell_idx.z <= last_max.z)
				return;

			grid.ForAllCellEntries(cell_idx, [&](unsigned int element_idx, const T& element)
			{
				float distAlongRay = -1;
				auto dist_data = clbDist(cell_idx, element_idx, element, distAlongRay, rad);
				if ((rayT <= distAlongRay || rayT <= minT * (1 + EPSILON)) && distAlongRay >= tmin  && distAlongRay <= tmax)//&& (distAlongRay < cellEndT || cellEndT >= maxT * (1.0f - EPSILON))
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
template<template<class> class Grid, typename T, typename F1, typename F2, typename F3> CUDA_FUNC_IN void TraverseGridBeam(const Ray& r, float tmin, float tmax, Grid<T>& grid, const F1& clbRad, const F2& clbDist, const F3& clbElement)
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