#pragma once

#include "Vector.h"
#include "Spectrum.h"

namespace CudaTracerLib {

class AlgebraHelper
{
public:
	CUDA_FUNC_IN static bool solveLinearSystem2x2(const float a[2][2], const float b[2], float x[2])
	{
		float det = a[0][0] * a[1][1] - a[0][1] * a[1][0];

		if (math::abs(det) <= RCPOVERFLOW)
			return false;

		float inverse = (float) 1.0f / det;

		x[0] = (a[1][1] * b[0] - a[0][1] * b[1]) * inverse;
		x[1] = (a[0][0] * b[1] - a[1][0] * b[0]) * inverse;

		return true;
	}

	CUDA_FUNC_IN static bool Quadratic(float A, float B, float C, float *t0, float *t1)
	{
		// Find quadratic discriminant
		float discrim = B * B - 4.f * A * C;
		if (discrim <= 0.) return false;
		float rootDiscrim = math::sqrt(discrim);

		// Compute quadratic _t_ values
		float q;
		if (B < 0) q = -.5f * (B - rootDiscrim);
		else       q = -.5f * (B + rootDiscrim);
		*t0 = q / A;
		*t1 = C / q;
		if (*t0 > *t1)
			swapk(*t0, *t1);
		return true;
	}

	//http://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
	// point p with respect to triangle (a, b, c)
	CUDA_FUNC_IN static bool Barycentric(const Vec3f& p, const Vec3f& a, const Vec3f& b, const Vec3f& c, float& u, float& v)
	{
		Vec3f v0 = b - a, v1 = c - a, v2 = p - a;
		float d00 = dot(v0, v0);
		float d01 = dot(v0, v1);
		float d11 = dot(v1, v1);
		float d20 = dot(v2, v0);
		float d21 = dot(v2, v1);
		float denom = d00 * d11 - d01 * d01;
		v = (d11 * d20 - d01 * d21) / denom;
		float w = (d00 * d21 - d01 * d20) / denom;
		u = 1.0f - v - w;
		return 0 <= v && v <= 1 && 0 <= u && u <= 1 && 0 <= w && w <= 1;
	}

	CUDA_FUNC_IN static float sqrDistanceToRay(const Ray& r, const Vec3f& pos, float& distanceAlongRay)
	{
		distanceAlongRay = dot(pos - r.ori(), r.dir());
		return distanceSquared(pos, r(distanceAlongRay));
	}

	CUDA_FUNC_IN static bool sphere_line_intersection(const Vec3f& p, float radSqr, const Ray& r, float& t_min, float& t_max)
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
};

}