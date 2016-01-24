#pragma once
#include <iostream>
#include "Vector.h"
#include "float4x4.h"

namespace CudaTracerLib {

struct Ray
{
private:
	Vec3f origin;
	Vec3f direction;
public:

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

	CUDA_FUNC_IN const Vec3f& ori() const
	{
		return origin;
	}

	CUDA_FUNC_IN Vec3f& ori()
	{
		return origin;
	}

	CUDA_FUNC_IN const Vec3f& dir() const
	{
		return direction;
	}

	CUDA_FUNC_IN Vec3f& dir()
	{
		return direction;
	}

	CUDA_FUNC_IN NormalizedT<Vec3f> dirUnit() const
	{
		return direction.normalized();
	}

	friend std::ostream& operator<< (std::ostream & os, const Ray& rhs)
	{
		os << "[" << rhs.origin << ", " << rhs.direction << "]";
		return os;
	}
};

}