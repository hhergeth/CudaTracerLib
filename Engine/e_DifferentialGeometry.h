#pragma once

#include <MathTypes.h>

struct DifferentialGeometry
{
	float3 P;
	Frame sys;
	float3 n;
	float3 dpdu, dpdv;
	float dudx, dudy, dvdx, dvdy;
	float2 uv[NUM_UV_SETS];
	float2 bary;
	unsigned char extraData;
	unsigned char hasUVPartials;
	unsigned char DUMMY[2];

	CUDA_FUNC_IN DifferentialGeometry() {}

	CUDA_DEVICE CUDA_HOST void computePartials(const Ray& r, const Ray& rx, const Ray& ry);

	CUDA_FUNC_IN float3 toWorld(const float3& v) const
	{
		return sys.toWorld(v);
	}

	CUDA_FUNC_IN float3 toLocal(const float3& v) const
	{
		return sys.toLocal(v);
	}
};