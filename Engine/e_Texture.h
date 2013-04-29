#pragma once

#include "cuda_runtime.h"
#include "..\Base\FileStream.h"
/*
class e_KernelTexture
{
private:
	cudaTextureObject_t obj;
public:
	e_KernelTexture() {}
	e_KernelTexture(cudaTextureObject_t _obj)
	{
		obj = _obj;
	}
	CUDA_ONLY_FUNC float4 Sample(float2& uv)
	{
#ifdef __CUDACC__
		return tex2D<float4>(obj, uv.x, 1.0f - uv.y); 
#else
		return make_float4(1,1,1,1);
#endif
	}
};*/

class CUDA_ALIGN(16) e_KernelTexture
{
public:
	RGBCOL* m_pDeviceData;
	float2 m_fDim;
	unsigned int m_uWidth;
public:
	CUDA_FUNC_IN float4 Sample(const float2& a_UV) const
	{
		float2 a_Coords = make_float2(frac(a_UV.x), frac(1.0f-a_UV.y));
		a_Coords = make_float2(m_fDim.x * a_Coords.x, m_fDim.y * a_Coords.y);
		unsigned int x = (unsigned int)a_Coords.x, y = (unsigned int)a_Coords.y;
		RGBCOL c = m_pDeviceData[y * m_uWidth + x];
		return COLORREFToFloat4(c);
	}
	template<typename T> CUDA_FUNC_IN T SampleT(const float2& a_UV) const
	{
		float2 a_Coords = make_float2(frac(a_UV.x), frac(1.0f-a_UV.y));
		a_Coords = make_float2(m_fDim.x * a_Coords.x, m_fDim.y * a_Coords.y);
		unsigned int x = (unsigned int)a_Coords.x, y = (unsigned int)a_Coords.y;
		T c = ((T*)m_pDeviceData)[y * m_uWidth + x];
		return c;
	}
};

class e_Texture
{
private:
	RGBCOL* m_pDeviceData;
	unsigned int m_uWidth;
	unsigned int m_uHeight;
	unsigned int m_uBpp;
	e_KernelTexture m_sKernelData;
public:
	e_Texture() {m_pDeviceData = 0; m_uWidth = m_uHeight = m_uBpp = -1;}
	e_Texture(float4& col);
	e_Texture(InputStream& a_In);
	void Free()
	{
		cudaFree(m_pDeviceData);
	}
	static void CompileToBinary(const char* a_InputFile, OutputStream& a_Out);
	e_KernelTexture CreateKernelTexture();
	e_KernelTexture getKernelData()
	{
		return m_sKernelData;
	}
	unsigned int getBufferSize()
	{
		return m_uWidth * m_uHeight * sizeof(RGBCOL);
	}
};

template<typename T> class e_TextureCreator
{
	int w, h;
	T* data;
public:
	e_TextureCreator(int _w, int _h)
	{
		w = _w;
		h = _h;
		data = new T[w * h];
	}
	~e_TextureCreator()
	{
		delete [] data;
	}
	void operator()(unsigned int x, unsigned int y, T& v)
	{
		data[y * w + x] = v;
	}
	void ToStream(OutputStream& a_Out)
	{
		a_Out << w;
		a_Out << h;
		a_Out << sizeof(T);
		a_Out.Write(data, w * h * sizeof(T));
	}
};