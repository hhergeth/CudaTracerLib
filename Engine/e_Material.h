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
	e_KernelMaterial(const std::string& name);
	CUDA_DEVICE CUDA_HOST bool SampleNormalMap(DifferentialGeometry& uv, const Vec3f& wi) const;
	CUDA_DEVICE CUDA_HOST float SampleAlphaMap(const DifferentialGeometry& uv) const;
	CUDA_DEVICE CUDA_HOST bool GetBSSRDF(const DifferentialGeometry& uv, const e_KernelBSSRDF** res) const;
	template<typename L> void LoadTextures(L& callback)
	{
		if(NormalMap.used && NormalMap.tex.Is<e_ImageTexture>())
			NormalMap.tex.As<e_ImageTexture>()->LoadTextures(callback);
		if (HeightMap.used && HeightMap.tex.Is<e_ImageTexture>())
			HeightMap.tex.As<e_ImageTexture>()->LoadTextures(callback);
		if (AlphaMap.used && AlphaMap.tex.Is<e_ImageTexture>())
			AlphaMap.tex.As<e_ImageTexture>()->LoadTextures(callback);
		std::vector<e_Texture*> T = bsdf.As()->getTextureList();
		for (size_t i = 0; i < T.size(); i++)
			if (T[i]->Is<e_ImageTexture>())
				T[i]->As<e_ImageTexture>()->LoadTextures(callback);
	}
	template<typename L> void UnloadTextures(L& callback)
	{
		if (NormalMap.used && NormalMap.tex.Is<e_ImageTexture>())
			NormalMap.tex.As<e_ImageTexture>()->UnloadTexture(callback);
		if (HeightMap.used && HeightMap.tex.Is<e_ImageTexture>())
			HeightMap.tex.As<e_ImageTexture>()->UnloadTexture(callback);
		if (AlphaMap.used && AlphaMap.tex.Is<e_ImageTexture>())
			AlphaMap.tex.As<e_ImageTexture>()->UnloadTexture(callback);
		std::vector<e_Texture*> T = bsdf.As()->getTextureList();
		for (size_t i = 0; i < T.size(); i++)
			if (T[i]->Is<e_ImageTexture>())
				T[i]->As<e_ImageTexture>()->UnloadTexture(callback);
	}
	void SetNormalMap(const e_Texture& tex)
	{
		if (HeightMap.used)
			throw std::runtime_error("Cannot set both height and normal map!");
		NormalMap.used = true;
		NormalMap.tex = tex;
	}
	void SetNormalMap(const std::string& path)
	{
		SetNormalMap(CreateTexture(path));
	}
	void SetHeightMap(const e_Texture& tex)
	{
		if(NormalMap.used)
			throw std::runtime_error("Cannot set both height and normal map!");
		HeightMap.used = true;
		HeightMap.tex = tex;
	}
	void SetHeightMap(const std::string& path)
	{
		SetHeightMap(CreateTexture(path));
	}
	void SetAlphaMap(const e_Texture& tex)
	{
		AlphaMap.used = true;
		AlphaMap.tex = tex;
	}
	void SetAlphaMap(const std::string& path)
	{
		SetAlphaMap(CreateTexture(path));
	}
	void setBssrdf(const Spectrum& sig_a, const Spectrum& sigp_s, float e);
};