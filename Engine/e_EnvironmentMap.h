#pragma once

#include "..\Math\vector.h"
#include "e_KernelTexture.h"

struct e_EnvironmentMap
{
public:
	e_KernelTexture<float3> Map;
	bool isSet;
public:

#ifdef __CUDACC__
	CUDA_DEVICE e_EnvironmentMap()
	{
	}
#else
	CUDA_HOST e_EnvironmentMap()
	{
		isSet = false;
	}
#endif

	static e_EnvironmentMap Identity()
	{
		e_EnvironmentMap m;
		m.isSet = false;
		m.Map = e_KernelTexture<float3>();
		return m;
	}

	e_EnvironmentMap(float3& f)
	{
		Map.SetData(e_KernelConstantTexture<float3>(f));
		isSet = true;
	}

	e_EnvironmentMap(const char* path)
	{
		Map.SetData(e_KernelImageTexture<float3>(CreateTextureMapping2D(e_KernelUVMapping2D()), path));
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
		MapParameters mp(make_float3(0), t, Onb());
		return Map.Evaluate(mp);
	}
};