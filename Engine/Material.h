#pragma once

#include "BSDF.h"
#include "Volumes.h"

namespace CudaTracerLib {

struct Material
{
public:
	struct mpHlp
	{
		int used;
		Texture tex;
		mpHlp()
			: used(0)
		{
		}
	};
	FixedString<64> Name;
	unsigned int NodeLightIndex;
	float HeightScale;
	float m_fAlphaThreshold;
	bool enableParallaxOcclusion;
	int parallaxMinSamples, parallaxMaxSamples;
	int usedBssrdf;
	VolumeRegion bssrdf;
	BSDFALL bsdf;
	mpHlp NormalMap;
	mpHlp HeightMap;
	mpHlp AlphaMap;
public:
	Material();
	Material(const std::string& name);
	CUDA_DEVICE CUDA_HOST bool SampleNormalMap(DifferentialGeometry& uv, const Vec3f& wi) const;
	CUDA_DEVICE CUDA_HOST float SampleAlphaMap(const DifferentialGeometry& uv) const;
	CUDA_DEVICE CUDA_HOST bool GetBSSRDF(const DifferentialGeometry& uv, const VolumeRegion** res) const;
	template<typename L> void LoadTextures(L& callback)
	{
		if (NormalMap.used && NormalMap.tex.Is<ImageTexture>())
			NormalMap.tex.As<ImageTexture>()->LoadTextures(callback);
		if (HeightMap.used && HeightMap.tex.Is<ImageTexture>())
			HeightMap.tex.As<ImageTexture>()->LoadTextures(callback);
		if (AlphaMap.used && AlphaMap.tex.Is<ImageTexture>())
			AlphaMap.tex.As<ImageTexture>()->LoadTextures(callback);
		std::vector<Texture*> T = bsdf.As()->getTextureList();
		for (size_t i = 0; i < T.size(); i++)
			if (T[i]->Is<ImageTexture>())
				T[i]->As<ImageTexture>()->LoadTextures(callback);
	}
	template<typename L> void UnloadTextures(L& callback)
	{
		if (NormalMap.used && NormalMap.tex.Is<ImageTexture>())
			NormalMap.tex.As<ImageTexture>()->UnloadTexture(callback);
		if (HeightMap.used && HeightMap.tex.Is<ImageTexture>())
			HeightMap.tex.As<ImageTexture>()->UnloadTexture(callback);
		if (AlphaMap.used && AlphaMap.tex.Is<ImageTexture>())
			AlphaMap.tex.As<ImageTexture>()->UnloadTexture(callback);
		std::vector<Texture*> T = bsdf.As()->getTextureList();
		for (size_t i = 0; i < T.size(); i++)
			if (T[i]->Is<ImageTexture>())
				T[i]->As<ImageTexture>()->UnloadTexture(callback);
	}
	void SetNormalMap(const Texture& tex)
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
	void SetHeightMap(const Texture& tex)
	{
		if (NormalMap.used)
			throw std::runtime_error("Cannot set both height and normal map!");
		HeightMap.used = true;
		HeightMap.tex = tex;
	}
	void SetHeightMap(const std::string& path)
	{
		SetHeightMap(CreateTexture(path));
	}
	void SetAlphaMap(const Texture& tex)
	{
		AlphaMap.used = true;
		AlphaMap.tex = tex;
	}
	void SetAlphaMap(const std::string& path)
	{
		SetAlphaMap(CreateTexture(path));
	}
};

}