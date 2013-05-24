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

enum e_KernelTexture_DataType
{
	vtRGBE,//possible return types : float3, float4
	vtRGBCOL,//possible return types : float3, float4
	vtGeneric,////possible return types : T
};

class CUDA_ALIGN(16) e_KernelTexture
{
public:
	RGBCOL* m_pDeviceData;
	float2 m_fDim;
	int2 m_uDim;
	unsigned int m_uWidth;
	e_KernelTexture_DataType m_uType;
public:
	template<typename T> CUDA_FUNC_IN T Sample(const float2& a_UV) const
	{
		return *at<T>(a_UV);
	}
	template<> CUDA_FUNC_IN float4 Sample(const float2& a_UV) const
	{
		if(m_uType == e_KernelTexture_DataType::vtGeneric)
			return *at<float4>(a_UV);
		else return make_float4(Sample<float3>(a_UV), 1);
	}
	template<> CUDA_FUNC_IN float3 Sample(const float2& a_UV) const
	{
		if(m_uType == e_KernelTexture_DataType::vtGeneric)
			return *at<float3>(a_UV);
		else
		{
			float2 q = make_float2(frac(a_UV.x), frac(1.0f-a_UV.y)) * m_fDim;
			int2 v = make_int2(q.x, q.y);
			float3 a = ld(v), b = ld(v + make_int2(1, 0)), c = ld(v + make_int2(0, 1)), d = ld(v + make_int2(1, 1));
			float3 e = lerp(a, b, frac(q.x)), f = lerp(c, d, frac(q.x));
			return lerp(e, f, frac(q.y));
		}
	}
	template<int W> CUDA_FUNC_IN void Gather(const float2& a_UV, float* data) const
	{
		float2 q = make_float2(frac(a_UV.x), frac(1.0f-a_UV.y)) * m_fDim;
		int2 v = make_int2(q.x, q.y);
		for(int i = -1; i < (W - 1); i++)
			for(int j = -1; j < (W - 1); j++)
			{
				float h;
				if(m_uType == e_KernelTexture_DataType::vtGeneric)
					h = *ld<float>(v, i, j);
				else if(m_uType == e_KernelTexture_DataType::vtRGBCOL)
					h = COLORREFToFloat3(*ld<RGBCOL>(v, i, j)).x;
				else if(m_uType == e_KernelTexture_DataType::vtRGBE)
					h = RGBEToFloat3(*ld<RGBE>(v, i, j)).x;
				data[(j + 1) * W + i + 1] = h;
			}
	}
private:
	template<typename T> CUDA_FUNC_IN T* ld(const int2& p2, int xo = 0, int yo = 0) const
	{
		int2 p = clamp(p2 + make_int2(xo, yo), make_int2(0,0), make_int2(m_fDim.x, m_fDim.y));
		return (T*)m_pDeviceData + p.y * m_uWidth + p.x;
	}
	template<typename T> CUDA_FUNC_IN T* at(const float2& a_UV) const
	{
		float2 a_Coords = make_float2(frac(a_UV.x), frac(1.0f-a_UV.y));
		a_Coords = make_float2(m_fDim.x * a_Coords.x, m_fDim.y * a_Coords.y);
		unsigned int x = (unsigned int)a_Coords.x, y = (unsigned int)a_Coords.y;
		return (T*)m_pDeviceData + y * m_uWidth + x;
	}
	CUDA_FUNC_IN float3 ld(const int2& uva) const
	{
		int2 uv = clamp(uva, make_int2(0, 0), m_uDim);
		float3 r;
		if(m_uType == e_KernelTexture_DataType::vtRGBCOL)
			r = COLORREFToFloat3(m_pDeviceData[uv.y * m_uWidth + uv.x]);
		else if(m_uType == e_KernelTexture_DataType::vtRGBE)
			r = RGBEToFloat3(((RGBE*)m_pDeviceData)[uv.y * m_uWidth + uv.x]);
		return r;
	}
};

class e_Texture
{
private:
	RGBCOL* m_pDeviceData;
	unsigned int m_uWidth;
	unsigned int m_uHeight;
	unsigned int m_uBpp;
	e_KernelTexture_DataType m_uType;
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
	static void CompileToBinary(const char* in, const char* out)
	{
		OutputStream o(out);
		CompileToBinary(in, o);
		o.Close();
	}
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