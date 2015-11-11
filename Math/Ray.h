#pragma once
#include <MathTypes.h>

namespace CudaTracerLib {

struct Ray
{
public:
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
	CUDA_FUNC_IN void NormalizeRayDirection(void) { direction = normalize(direction); }

	CUDA_FUNC_IN Ray operator *(const float4x4& m) const
	{
		return Ray(m.TransformPoint(origin), m.TransformDirection(direction));
	}
	CUDA_FUNC_IN Vec3f operator()(float d) const
	{
		return origin + d * direction;
	}

	friend std::ostream& operator<< (std::ostream & os, const Ray& rhs)
	{
		os << "[" << rhs.origin << ", " << rhs.direction << "]";
		return os;
	}
};


}