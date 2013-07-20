#pragma once

#include "..\Math\vector.h"
#include "e_Sampler.h"

struct e_EnvironmentMap
{
private:
	e_Sampler<float3> Map;
	bool isSet;
public:
	CUDA_DEVICE e_EnvironmentMap()
	{
	}

	static e_EnvironmentMap Identity()
	{
		e_EnvironmentMap m;
		m.isSet = false;
		m.Map = e_Sampler<float3>(make_float3(0));
		return m;
	}

	e_EnvironmentMap(float3& f)
		: Map(f)
	{
		isSet = true;
	}

	e_EnvironmentMap(const char* path)
		: Map(path, true)
	{
		isSet = true;
	}

	template<typename L> void LoadTextures(L callback)
	{
		Map.LoadTextures(callback);
	}

	CUDA_FUNC_IN bool CanSample()
	{
		return isSet;
	}

	CUDA_FUNC_IN float3 Sample(Ray& r)
	{
		float3 n = normalize(r.direction);
		//float2 t = make_float2(asinf(n.x) * INV_PI + 0.5f, asinf(n.y) * INV_PI + 0.5f);
		float2 t = make_float2(0.5f + atan2(n.z, n.x) * 0.5f * INV_PI, 0.5f - asin(n.y) * INV_PI);
		return Map.Sample(t);
	}
};