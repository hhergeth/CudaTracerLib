#pragma once

#include <Math/Vector.h>
#include <Math/Frame.h>

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
};

}