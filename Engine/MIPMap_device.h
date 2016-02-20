#pragma once
#include <Math/Vector.h>

namespace CudaTracerLib {

struct Spectrum;

#define MAX_MIPS 16
#define MTS_MIPMAP_LUT_SIZE 64

enum ImageWrap
{
	TEXTURE_REPEAT,
	TEXTURE_CLAMP,
	TEXTURE_MIRROR,
	TEXTURE_BLACK,
};

enum ImageFilter
{
	TEXTURE_Point,
	TEXTURE_Bilinear,
	TEXTURE_Anisotropic,
	TEXTURE_Trilinear,
};

enum Texture_DataType
{
	vtRGBE,
	vtRGBCOL,
};

CUDA_FUNC_IN bool WrapCoordinates(const Vec2f& a_UV, const Vec2f& dim, ImageWrap w, Vec2f* loc)
{
	switch (w)
	{
	case TEXTURE_REPEAT:
		*loc = Vec2f(math::frac(a_UV.x), math::frac(1.0f - a_UV.y)) * dim;
		return true;
	case TEXTURE_CLAMP:
		*loc = Vec2f(math::clamp01(a_UV.x), math::clamp01(1.0f - a_UV.y)) * dim;
		return true;
	case TEXTURE_MIRROR:
		loc->x = (int)a_UV.x % 2 == 0 ? math::frac(a_UV.x) : 1.0f - math::frac(a_UV.x);
		loc->y = (int)a_UV.x % 2 == 0 ? math::frac(a_UV.y) : 1.0f - math::frac(a_UV.y);
		*loc = *loc * dim;
		return true;
	case TEXTURE_BLACK:
		if (a_UV.x < 0 || a_UV.x >= 1 || a_UV.y < 0 || a_UV.y >= 1)
			return false;
		*loc = a_UV * dim;
		return true;
	}
	return false;
}

struct KernelMIPMap
{
	CUDA_ALIGN(16) unsigned int* m_pDeviceData;
	CUDA_ALIGN(16) unsigned int* m_pHostData;
	CUDA_ALIGN(16) Vec2f m_fDim;
	unsigned int m_uWidth, m_uHeight;
	Texture_DataType m_uType;
	ImageWrap m_uWrapMode;
	ImageFilter m_uFilterMode;
	unsigned int m_sOffsets[MAX_MIPS];
	unsigned int m_uLevels;
	float m_weightLut[MTS_MIPMAP_LUT_SIZE];

	//Texture functions
	CTL_EXPORT CUDA_DEVICE CUDA_HOST Spectrum Sample(const Vec2f& uv) const;
	CTL_EXPORT CUDA_DEVICE CUDA_HOST float SampleAlpha(const Vec2f& uv) const;
	CTL_EXPORT CUDA_DEVICE CUDA_HOST void evalGradient(const Vec2f& uv, Spectrum* gradient) const;
	//MipMap functions
	CTL_EXPORT CUDA_DEVICE CUDA_HOST Spectrum Sample(const Vec2f& a_UV, float width) const;
	CTL_EXPORT CUDA_DEVICE CUDA_HOST Spectrum Sample(float width, int x, int y) const;
	CTL_EXPORT CUDA_DEVICE CUDA_HOST Spectrum eval(const Vec2f& uv, const Vec2f& d0, const Vec2f& d1) const;
private:
	CTL_EXPORT CUDA_DEVICE CUDA_HOST Spectrum Texel(unsigned int level, const Vec2f& a_UV) const;
	CTL_EXPORT CUDA_DEVICE CUDA_HOST Spectrum triangle(unsigned int level, const Vec2f& a_UV) const;
	CTL_EXPORT CUDA_DEVICE CUDA_HOST Spectrum evalEWA(unsigned int level, const Vec2f &uv, float A, float B, float C) const;
};

}