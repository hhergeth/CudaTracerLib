#pragma once

#include "e_BSDF.h"

struct e_KernelBSSRDF
{
	CUDA_FUNC_IN e_KernelBSSRDF()
	{
	}

	CUDA_FUNC_IN e_KernelBSSRDF(float _e, const Spectrum& sa, const Spectrum& sps)
	{
		e = _e;
		sig_a = sa;
		sigp_s = sps;
	}

    float e;
    Spectrum sig_a, sigp_s;
};

struct e_KernelMaterial
{
public:
	struct mpHlp
	{
		int used;
		e_Texture tex;
		mpHlp()
			: used(0)
		{
		}
	};
	e_KernelBSSRDF bssrdf;
	FixedString<64> Name;
	unsigned int NodeLightIndex;
	float HeightScale;
	float m_fAlphaThreshold;
	bool enableParallaxOcclusion;
	int parallaxMinSamples, parallaxMaxSamples;
	int usedBssrdf;
	BSDFALL bsdf;
	mpHlp NormalMap;
	mpHlp HeightMap;
	mpHlp AlphaMap;
public:
	e_KernelMaterial();
	e_KernelMaterial(const char* name);
	CUDA_DEVICE CUDA_HOST bool SampleNormalMap(DifferentialGeometry& uv, const Vec3f& wi) const;
	CUDA_DEVICE CUDA_HOST float SampleAlphaMap(const DifferentialGeometry& uv) const;
	CUDA_DEVICE CUDA_HOST bool GetBSSRDF(const DifferentialGeometry& uv, const e_KernelBSSRDF** res) const;
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
	void SetNormalMap(const e_Texture& tex)
	{
		if(HeightMap.used)
			throw 1;
		NormalMap.used = true;
		NormalMap.tex = tex;
	}
	void SetNormalMap(const char* path)
	{
		SetNormalMap(CreateTexture(path));
	}
	void SetHeightMap(const e_Texture& tex)
	{
		if(NormalMap.used)
			throw 1;
		HeightMap.used = true;
		HeightMap.tex = tex;
	}
	void SetHeightMap(const char* path)
	{
		SetHeightMap(CreateTexture(path));
	}
	void SetAlphaMap(const e_Texture& tex)
	{
		if(AlphaMap.used)
			throw 1;
		AlphaMap.used = true;
		AlphaMap.tex = tex;
	}
	void SetAlphaMap(const char* path)
	{
		SetAlphaMap(CreateTexture(path));
	}
	void setBssrdf(const Spectrum& sig_a, const Spectrum& sigp_s, float e);
};