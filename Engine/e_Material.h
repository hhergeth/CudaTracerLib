#pragma once

#include "e_BSDF.h"

struct e_KernelBSSRDF
{
	CUDA_FUNC_IN e_KernelBSSRDF()
	{
	}

	CUDA_FUNC_IN e_KernelBSSRDF(float _e, float3& sa, float3& sps)
	{
		e = _e;
		sig_a = sa;
		sigp_s = sps;
	}

    float e;
    float3 sig_a, sigp_s;
};

struct e_KernelMaterial
{
public:
	template<typename T> struct mpHlp
	{
		bool used;
		e_KernelTexture<T> tex;
		mpHlp()
			: used(false)
		{
		}
	};
	BSDFALL bsdf;
	e_KernelBSSRDF bssrdf;
	bool usedBssrdf;
	mpHlp<float3> NormalMap;
	mpHlp<float3> HeightMap;
	mpHlp<float4> AlphaMap;
	e_String Name;
	unsigned int NodeLightIndex;
	float HeightScale;
	float m_fAlphaThreshold;
public:
	e_KernelMaterial(char* name = 0);
	CUDA_DEVICE CUDA_HOST bool SampleNormalMap(const MapParameters& uv, float3* normal) const;
	CUDA_DEVICE CUDA_HOST float SampleAlphaMap(const MapParameters& uv) const;
	CUDA_DEVICE CUDA_HOST bool GetBSSRDF(const MapParameters& uv, const e_KernelBSSRDF* res) const;
	template<typename L> void LoadTextures(L callback)
	{
		if(NormalMap.used)
			NormalMap.tex.LoadTextures(callback);
		if(HeightMap.used)
			HeightMap.tex.LoadTextures(callback);
		if(AlphaMap.used)
			AlphaMap.tex.LoadTextures(callback);
		bsdf.LoadTextures(callback);
	}
	void SetNormalMap(const e_KernelTexture<float3>& tex)
	{
		if(HeightMap.used)
			throw 1;
		NormalMap.used = true;
		NormalMap.tex = tex;
	}
	void SetNormalMap(const char* path)
	{
		SetNormalMap(CreateTexture(path, make_float3(0)));
	}
	void SetHeightMap(const e_KernelTexture<float3>& tex)
	{
		if(NormalMap.used)
			throw 1;
		HeightMap.used = true;
		HeightMap.tex = tex;
	}
	void SetHeightMap(const char* path)
	{
		SetHeightMap(CreateTexture(path, make_float3(0)));
	}
	void SetAlphaMap(const e_KernelTexture<float4>& tex)
	{
		if(AlphaMap.used)
			throw 1;
		AlphaMap.used = true;
		AlphaMap.tex = tex;
	}
	void SetAlphaMap(const char* path)
	{
		SetAlphaMap(CreateTexture(path, make_float4(0)));
	}
};