#pragma once
#include <MathTypes.h>

class Ray
{
public:  //Data

  // basic information about a ray
  float3 origin;
  float3 direction;
  
public:  // Methods
	CUDA_FUNC Ray()
	{
	}
	// Set up the ray with a give origin and direction (and optionally a ray depth)
	CUDA_FUNC_IN Ray( const float3 &orig, const float3 &dir )
		: origin(orig), direction(dir)
	{
	}
	// Normalize ray direction
	CUDA_FUNC_IN void NormalizeRayDirection( void ) { direction = normalize(direction); }	

	CUDA_FUNC_IN float3 getOrigin() const{return origin;}
	CUDA_FUNC_IN float3 getDirection() const{return direction;}
	CUDA_FUNC_IN Ray operator *(const float4x4& m) const
	{
		return Ray(m * origin, m.TransformNormal(direction));
	}
	CUDA_FUNC_IN float3 operator()(float d) const
	{
		return origin + d * direction;
	}
};

