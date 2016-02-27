#pragma once

#include "BSDF.h"
#include "Volumes.h"

namespace CudaTracerLib {

//Luminance : sample().getLuminance() >= test_val_scalar
//Alpha : sample().alpha >= test_val_scalar, this mode is only applicable for ImageTextures
//Color : (sample() - test_val_color).abs().max() <= test_val_scalar
//not using flags prevents inconsistent states which would have to be checked during execution
enum AlphaBlendState
{
	Disabled = 0,
	//prefix(0 = alpha, 4 = refl) | type
	AlphaMap_Luminance = 0 | 1,
	AlphaMap_Alpha = 0 | 2,
	AlphaMap_Color = 0 | 3,
	ReflectanceMap_Luminance = 4 | 1,
	ReflectanceMap_Alpha = 4 | 2,
	ReflectanceMap_Color = 4 | 3,
};

struct AlphaBlendData
{
	AlphaBlendState state;
	Texture tex;
	float test_val_scalar;
	Spectrum test_val_color;

	CUDA_FUNC_IN bool used() const
	{
		return state != AlphaBlendState::Disabled;
	}
};

struct TriangleData;
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
	bool enableParallaxOcclusion;
	int parallaxMinSamples, parallaxMaxSamples;
	int usedBssrdf;
	VolumeRegion bssrdf;
	BSDFALL bsdf;
	mpHlp NormalMap;
	mpHlp HeightMap;
	AlphaBlendData AlphaMap;
public:
	CTL_EXPORT Material();
	CTL_EXPORT Material(const std::string& name);
	CTL_EXPORT CUDA_DEVICE CUDA_HOST bool SampleNormalMap(DifferentialGeometry& uv, const Vec3f& wi) const;
	CTL_EXPORT CUDA_DEVICE CUDA_HOST bool AlphaTest(const Vec2f& bary, const Vec2f& uv) const;
	CTL_EXPORT CUDA_DEVICE CUDA_HOST bool GetBSSRDF(const DifferentialGeometry& uv, const VolumeRegion** res) const;
	template<typename L> void LoadTextures(L& callback)
	{
		if (NormalMap.used && NormalMap.tex.Is<ImageTexture>())
			NormalMap.tex.As<ImageTexture>()->LoadTextures(callback);
		if (HeightMap.used && HeightMap.tex.Is<ImageTexture>())
			HeightMap.tex.As<ImageTexture>()->LoadTextures(callback);
		if (AlphaMap.used() && AlphaMap.tex.Is<ImageTexture>())
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
		if (AlphaMap.used() && AlphaMap.tex.Is<ImageTexture>())
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
	void SetHeightMap(const Texture& tex)
	{
		if (NormalMap.used)
			throw std::runtime_error("Cannot set both height and normal map!");
		HeightMap.used = true;
		HeightMap.tex = tex;
	}
	void SetAlphaMap(const Texture& tex, AlphaBlendState state)
	{
		AlphaMap.state = state;
		AlphaMap.tex = tex;
	}
};

}