#pragma once

#include "cuda_runtime.h"
#include "..\Base\FileStream.h"

#define max_MIPS 16
#define MTS_MIPMAP_LUT_SIZE 64

enum e_ImageWrap
{
	TEXTURE_REPEAT,
	TEXTURE_CLAMP,
	TEXTURE_MIRROR,
	TEXTURE_BLACK,
};

enum e_ImageFilter
{
	TEXTURE_Point,
	TEXTURE_Bilinear,
	TEXTURE_Anisotropic
};

enum e_KernelTexture_DataType
{
	vtRGBE,
	vtRGBCOL,
};

CUDA_FUNC_IN bool WrapCoordinates(const Vec2f& a_UV, const Vec2f& dim, e_ImageWrap w, Vec2f* loc)
{
	switch(w)
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
		if (a_UV.x < 0 || a_UV.x >= 1|| a_UV.y < 0 || a_UV.y >= 1)
			return false;
		*loc = a_UV * dim;
		return true;
	}
	return false;
}

struct e_KernelMIPMap
{
	CUDA_ALIGN(16) unsigned int* m_pDeviceData;
	CUDA_ALIGN(16) unsigned int* m_pHostData;
	CUDA_ALIGN(16) Vec2f m_fDim;
	unsigned int m_uWidth, m_uHeight;
	e_KernelTexture_DataType m_uType;
	e_ImageWrap m_uWrapMode;
	e_ImageFilter m_uFilterMode;
	unsigned int m_sOffsets[max_MIPS];
	unsigned int m_uLevels;
	float m_weightLut[MTS_MIPMAP_LUT_SIZE];

	//Texture functions
	CUDA_DEVICE CUDA_HOST Spectrum Sample(const Vec2f& uv) const;
	CUDA_DEVICE CUDA_HOST float SampleAlpha(const Vec2f& uv) const;
	CUDA_DEVICE CUDA_HOST void evalGradient(const Vec2f& uv, Spectrum* gradient) const;
	//MipMap functions
	CUDA_DEVICE CUDA_HOST Spectrum Sample(const Vec2f& a_UV, float width) const;
	CUDA_DEVICE CUDA_HOST Spectrum Sample(float width, int x, int y) const;
	CUDA_DEVICE CUDA_HOST Spectrum eval(const Vec2f& uv, const Vec2f& d0, const Vec2f& d1) const;
private:
	CUDA_DEVICE CUDA_HOST Spectrum Texel(unsigned int level, const Vec2f& a_UV) const;
	CUDA_DEVICE CUDA_HOST Spectrum triangle(unsigned int level, const Vec2f& a_UV) const;
	CUDA_DEVICE CUDA_HOST Spectrum evalEWA(unsigned int level, const Vec2f &uv, float A, float B, float C) const;
};

class e_MIPMap
{
	unsigned int* m_pDeviceData;
	unsigned int* m_pHostData;
	unsigned int m_uWidth;
	unsigned int m_uHeight;
	unsigned int m_uBpp;
	unsigned int m_uLevels;
	unsigned int m_uSize;
	e_KernelTexture_DataType m_uType;
	e_ImageWrap m_uWrapMode;
	unsigned int m_sOffsets[max_MIPS];
	float m_weightLut[MTS_MIPMAP_LUT_SIZE];
public:
	e_ImageFilter m_uFilterMode;
	e_MIPMap() {m_pDeviceData = 0; m_uWidth = m_uHeight = m_uBpp = 0xffffffff;}
	e_MIPMap(InputStream& a_In);
	void Free()
	{
		CUDA_FREE(m_pDeviceData);
		free(m_pHostData);
	}
	static void CompileToBinary(const char* a_InputFile, OutputStream& a_Out, bool a_MipMap);
	static void CompileToBinary(const char* in, const char* out, bool a_MipMap)
	{
		OutputStream o(out);
		CompileToBinary(in, o, a_MipMap);
		o.Close();
	}
	static void CreateSphericalSkydomeTexture(const char* front, const char* back, const char* left, const char* right, const char* top, const char* bottom, const char* outFile);
	e_KernelMIPMap getKernelData();
	unsigned int getNumMips()
	{
		return m_uLevels;
	}
	unsigned int getBufferSize()
	{
		return m_uSize;
	}
};