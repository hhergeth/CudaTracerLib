#pragma once

#include <Math/Vector.h>
#include <Math/Frame.h>
#include <Math/AABB.h>

namespace CudaTracerLib {

struct Ray;

struct DifferentialGeometry
{
	Vec3f P;
	Frame sys;
	NormalizedT<Vec3f> n;
	Vec3f dpdu, dpdv;
	float dudx, dudy, dvdx, dvdy;
	Vec2f uv[NUM_UV_SETS];
	Vec2f bary;
	unsigned char extraData;
	unsigned char hasUVPartials;
	unsigned char DUMMY[2];

	CUDA_FUNC_IN DifferentialGeometry() {}

	CTL_EXPORT CUDA_DEVICE CUDA_HOST void computePartials(const Ray& r, const Ray& rx, const Ray& ry);

	CUDA_FUNC_IN NormalizedT<Vec3f> toWorld(const NormalizedT<Vec3f>& v) const
	{
		return sys.toWorld(v);
	}

	CUDA_FUNC_IN NormalizedT<Vec3f> toLocal(const NormalizedT<Vec3f>& v) const
	{
		return sys.toLocal(v);
	}

	CUDA_FUNC_IN AABB ComputeOnSurfaceDiskBounds(float rad) const
	{
		Vec3f a = rad*(-sys.t - sys.s) + P, b = rad*(sys.t - sys.s) + P, c = rad*(-sys.t + sys.s) + P, d = rad*(sys.t + sys.s) + P;
		return AABB(min(a, b, c, d), max(a, b, c, d));
	}
};

}