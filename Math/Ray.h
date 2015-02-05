#pragma once
#include <MathTypes.h>

class Ray
{
public:  //Data

  // basic information about a ray
  Vec3f origin;
  Vec3f direction;
  
public:  // Methods
	CUDA_FUNC Ray()
	{
	}
	// Set up the ray with a give origin and direction (and optionally a ray depth)
	CUDA_FUNC_IN Ray(const Vec3f &orig, const Vec3f &dir)
		: origin(orig), direction(dir)
	{
	}
	// Normalize ray direction
	CUDA_FUNC_IN void NormalizeRayDirection( void ) { direction = normalize(direction); }	

	CUDA_FUNC_IN Vec3f getOrigin() const{return origin;}
	CUDA_FUNC_IN Vec3f getDirection() const{return direction;}
	CUDA_FUNC_IN Ray operator *(const float4x4& m) const
	{
		return Ray(m.TransformPoint(origin), m.TransformDirection(direction));
	}
	CUDA_FUNC_IN Vec3f operator()(float d) const
	{
		return origin + d * direction;
	}
};

