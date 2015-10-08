#pragma once
#include "../../Defines.h"
#include "../../MathTypes.h"

struct k_Beam
{
	Vec3f pos;
	Vec3f dir;
	float t;
	Spectrum Phi;
	unsigned int lastEntry;
	k_Beam(){}
	CUDA_FUNC_IN k_Beam(const Vec3f& p, const Vec3f& d, float t, const Spectrum& ph)
		: pos(p), dir(d), t(t), Phi(ph), lastEntry(0)
	{

	}
};