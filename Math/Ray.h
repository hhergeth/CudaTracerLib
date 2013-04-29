#pragma once
#include "Vector.h"

class FlattendedTriangle;

struct HitInfo
{
	// Information about the hit point
	float t;//distance along the ray
	float2 uv;
private:
	unsigned int object;//object to be shaded	
public:
	CUDA_ONLY_FUNC void init()
	{
		t = FLT_MAX;
		object = 0;
	}
	CUDA_ONLY_FUNC void setHit(unsigned int o, float f, float2& _uv)
	{
		object = o;
		uv = _uv;
		t = f;
	}
	CUDA_ONLY_FUNC unsigned int hasHit()
	{
		return object != -1;
	}
	CUDA_ONLY_FUNC unsigned int getHit()
	{
		return object;
	}
};


