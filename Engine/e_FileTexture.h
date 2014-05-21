#pragma once

#include "cuda_runtime.h"
#include "..\Base\FileStream.h"

#define MAX_MIPS 16

enum e_ImageWrap
{
	TEXTURE_REPEAT,
	TEXTURE_CLAMP,
	TEXTURE_MIRROR,
	TEXTURE_BLACK,
};

enum e_KernelTexture_DataType
{
	vtRGBE,
	vtRGBCOL,
};

CUDA_FUNC_IN bool WrapCoordinates(const float2& a_UV, const float2& dim, e_ImageWrap w, float2* loc)
{
	switch(w)
	{
	case TEXTURE_REPEAT:
		*loc = make_float2(frac(a_UV.x), frac(1.0f-a_UV.y)) * dim;
		return true;
	case TEXTURE_CLAMP:
		*loc = make_float2(clamp01(a_UV.x), clamp01(1.0f-a_UV.y)) * dim;
		return true;
	case TEXTURE_MIRROR:
		loc->x = (int)a_UV.x % 2 == 0 ? frac(a_UV.x) : 1.0f - frac(a_UV.x);
		loc->y = (int)a_UV.x % 2 == 0 ? frac(a_UV.y) : 1.0f - frac(a_UV.y);
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
	CUDA_ALIGN(16) float2 m_fDim;
	CUDA_ALIGN(16) int2 m_uDim;
	unsigned int m_uWidth, m_uHeight;
	e_KernelTexture_DataType m_uType;
	e_ImageWrap m_uWrapMode;
	unsigned int m_sOffsets[MAX_MIPS];
	unsigned int m_uLevels;

	//Texture functions
	CUDA_DEVICE CUDA_HOST Spectrum Sample(const float2& uv) const;
	CUDA_DEVICE CUDA_HOST float SampleAlpha(const float2& uv) const;
	//MipMap functions
	CUDA_DEVICE CUDA_HOST Spectrum Sample(const float2& a_UV, float width) const;
	CUDA_DEVICE CUDA_HOST Spectrum Sample(float width, int x, int y) const;
private:
	CUDA_DEVICE CUDA_HOST Spectrum Texel(unsigned int level, const float2& a_UV) const;
	CUDA_DEVICE CUDA_HOST Spectrum triangle(unsigned int level, const float2& a_UV) const;
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
	e_KernelMIPMap m_sKernelData;
	unsigned int m_sOffsets[MAX_MIPS];
public:
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
	e_KernelMIPMap CreateKernelTexture();
	e_KernelMIPMap getKernelData()
	{
		return m_sKernelData;
	}
	unsigned int getNumMips()
	{
		return m_uLevels;
	}
	unsigned int getBufferSize()
	{
		return m_uSize;
	}
};