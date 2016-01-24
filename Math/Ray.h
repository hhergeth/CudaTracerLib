#pragma once
#include <MathTypes.h>

namespace CudaTracerLib {

struct Ray
{
	Vec3f origin;
	Vec3f direction;

	CUDA_FUNC Ray()
	{
	}
	CUDA_FUNC_IN Ray(const Vec3f &orig, const Vec3f &dir)
		: origin(orig), direction(dir)
	{
	}

	CUDA_FUNC_IN Ray operator *(const float4x4& m) const
	{
		return Ray(m.TransformPoint(origin), m.TransformDirection(direction));
	}
	CUDA_FUNC_IN Vec3f operator()(float d) const
	{
		return origin + d * direction;
	}

	CUDA_FUNC_IN NormalizedT<Vec3f> dir() const
	{
		return direction.normalized();
	}

	friend std::ostream& operator<< (std::ostream & os, const Ray& rhs)
	{
		os << "[" << rhs.origin << ", " << rhs.direction << "]";
		return os;
	}
};
/*
namespace __ray_internal__
{
	typedef std::enable_if<std::is_same<int, int>::value>::type F;
}

template<> struct NormalizedT<Ray, __ray_internal__::F> : Ray
{
	CUDA_FUNC NormalizedT<Ray, __ray_internal__::F>()
	{
	}
	CUDA_FUNC NormalizedT<Ray, __ray_internal__::F>(const Ray& r)
		: Ray(r.origin, r.direction.normalized())
	{
	}
	CUDA_FUNC_IN NormalizedT<Ray, __ray_internal__::F>(const Vec3f &o, const NormalizedT<Vec3f> &d)
		: Ray(o, d)
	{
	}

	CUDA_FUNC_IN const NormalizedT<Vec3f>& dir() const
	{
		return *(const NormalizedT<Vec3f>*)&direction;
	}

	CUDA_FUNC_IN NormalizedT<Vec3f>& dir()
	{
		return *(NormalizedT<Vec3f>*)&direction;
	}
};
*/
}