#pragma once

#include <MathTypes.h>

struct DifferentialGeometry
{
	Vec3f P;
	Frame sys;
	Vec3f n;
	Vec3f dpdu, dpdv;
	float dudx, dudy, dvdx, dvdy;
	Vec2f uv[NUM_UV_SETS];
	Vec2f bary;
	unsigned char extraData;
	unsigned char hasUVPartials;
	unsigned char DUMMY[2];

	CUDA_FUNC_IN DifferentialGeometry() {}

	CUDA_DEVICE CUDA_HOST void computePartials(const Ray& r, const Ray& rx, const Ray& ry);

	CUDA_FUNC_IN Vec3f toWorld(const Vec3f& v) const
	{
		return sys.toWorld(v);
	}

	CUDA_FUNC_IN Vec3f toLocal(const Vec3f& v) const
	{
		return sys.toLocal(v);
	}
};