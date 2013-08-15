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

enum e_ImageWrap
{
	TEXTURE_REPEAT,
	TEXTURE_CLAMP,
	TEXTURE_MIRROR,
	TEXTURE_BLACK,
};

enum e_KernelTexture_DataType
{
	vtRGBE,//possible return types : float3, float4
	vtRGBCOL,//possible return types : float3, float4
	vtGeneric,////possible return types : T
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
}

class CUDA_ALIGN(16) e_KernelFileTexture
{
public:
	CUDA_ALIGN(16) void* m_pDeviceData;
	CUDA_ALIGN(16) float2 m_fDim;
	CUDA_ALIGN(16) int2 m_uDim;
	unsigned int m_uWidth;
	e_KernelTexture_DataType m_uType;
	e_ImageWrap m_uWrapMode;
public:
	template<typename T> CUDA_FUNC_IN T Sample(const float2& a_UV) const
	{
		return *at<T>(a_UV);
	}
	template<> CUDA_FUNC_IN float4 Sample(const float2& a_UV) const
	{
		if(m_uType == e_KernelTexture_DataType::vtGeneric)
			return *at<float4>(a_UV);
		else
		{
			float2 q = make_float2(frac(a_UV.x), frac(1.0f-a_UV.y)) * m_fDim;
			int2 v = make_int2(q.x, q.y);
			float4 a = ld(v), b = ld(v + make_int2(1, 0)), c = ld(v + make_int2(0, 1)), d = ld(v + make_int2(1, 1));
			float4 e = lerp(a, b, frac(q.x)), f = lerp(c, d, frac(q.x));
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
	template<> CUDA_FUNC_IN float3 Sample(const float2& a_UV) const
	{
		if(m_uType == e_KernelTexture_DataType::vtGeneric)
			return *at<float3>(a_UV);
		else return make_float3(Sample<float4>(a_UV));
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
	CUDA_FUNC_IN float4 ld(const int2& uva) const
	{
		int2 uv = clamp(uva, make_int2(0, 0), m_uDim);
		float4 r;
		if(m_uType == e_KernelTexture_DataType::vtRGBCOL)
			r = COLORREFToFloat4(((RGBCOL*)m_pDeviceData)[uv.y * m_uWidth + uv.x]);
		else if(m_uType == e_KernelTexture_DataType::vtRGBE)
			r = make_float4(RGBEToFloat3(((RGBE*)m_pDeviceData)[uv.y * m_uWidth + uv.x]), 1);
		return r;
	}
};

class e_FileTexture
{
private:
	void* m_pDeviceData;
	unsigned int m_uWidth;
	unsigned int m_uHeight;
	unsigned int m_uBpp;
	e_KernelTexture_DataType m_uType;
	e_KernelFileTexture m_sKernelData;
	e_ImageWrap m_uWrapMode;
public:
	e_FileTexture() {m_pDeviceData = 0; m_uWidth = m_uHeight = m_uBpp = -1; m_uWrapMode = TEXTURE_REPEAT;}
	e_FileTexture(float4& col);
	e_FileTexture(InputStream& a_In);
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
	e_KernelFileTexture CreateKernelTexture();
	e_KernelFileTexture getKernelData()
	{
		return m_sKernelData;
	}
	unsigned int getBufferSize()
	{
		return m_uWidth * m_uHeight * m_uBpp;
	}
	e_ImageWrap getWrapMode()
	{
		return m_uWrapMode;
	}
	void setWrapMode(e_ImageWrap& w)
	{
		m_uWrapMode = w;
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

#define MAX_MIPS 16
struct e_KernelMIPMap
{
	void* m_pDeviceData;
	unsigned int m_uWidth;
	unsigned int m_uHeight;
	unsigned int m_uLevels;
	e_KernelTexture_DataType m_uType;
	e_ImageWrap m_uWrapMode;
	unsigned int m_sOffsets[MAX_MIPS];

	template<typename T> CUDA_FUNC_IN T Sample(const float2& a_UV, float width) const
	{
		struct fu
		{
			CUDA_FUNC_IN T operator()(const void* v, unsigned int o, e_KernelTexture_DataType t)
			{
				return ((T*)v)[o];
			}
		};
		return Sample(a_UV, width, fu());
	}
	template<> CUDA_FUNC_IN float4 Sample(const float2& a_UV, float width) const
	{
		struct fu
		{
			CUDA_FUNC_IN float4 operator()(const void* v, unsigned int o, e_KernelTexture_DataType t)
			{
				switch(t)
				{
				case vtRGBE:
					return make_float4(RGBEToFloat3(((RGBE*)v)[o]));
				case vtRGBCOL:
					return COLORREFToFloat4(((RGBCOL*)v)[o]);
				default:
					return ((float4*)v)[o];
				}
				return make_float4(0);
			}
		};
		return Sample<float4, fu>(a_UV, width, fu());
	}
	template<> CUDA_FUNC_IN float3 Sample(const float2& a_UV, float width) const
	{
		struct fu
		{
			CUDA_FUNC_IN float3 operator()(const void* v, unsigned int o, e_KernelTexture_DataType t)
			{
				switch(t)
				{
				case vtRGBE:
					return RGBEToFloat3(((RGBE*)v)[o]);
				case vtRGBCOL:
					return COLORREFToFloat3(((RGBCOL*)v)[o]);
				default:
					return ((float3*)v)[o];
				}
				return make_float3(0);
			}
		};
		return Sample<float3, fu>(a_UV, width, fu());
	}
private:
	template<typename T, typename F> CUDA_FUNC_IN T Texel(F func, unsigned int level, const float2& a_UV) const
	{
		float2 l;
		if(!WrapCoordinates(a_UV, make_float2(m_uWidth >> level, m_uHeight >> level), m_uWrapMode, &l))
			return T();
		else
		{
			unsigned int x = (unsigned int)l.x, y = (unsigned int)l.y;
			return func(m_pDeviceData, m_sOffsets[level] + y * (m_uWidth >> level) + x, m_uType);
		}
	}
	template<typename T, typename F> CUDA_FUNC_IN T Sample(const float2& a_UV, float width, F func) const
	{
		float level = m_uLevels - 1 + Log2(MAX((float)width, 1e-8f));
		if (level < 0)
			return triangle<T, F>(func, 0, a_UV);
		else if (level >= m_uLevels - 1)
			return Texel<T, F>(func, m_uLevels - 1, a_UV);
		else
		{
			int iLevel = Floor2Int(level);
			float delta = level - iLevel;
			return (1.f-delta) * triangle<T, F>(func, iLevel, a_UV) + delta * triangle<T, F>(func, iLevel+1, a_UV);
		}
	}
	template<typename T, typename F> CUDA_FUNC_IN T triangle(F func, unsigned int level, const float2& a_UV) const
	{
		level = clamp(level, 0u, m_uLevels-1);
		float2 s = make_float2(m_uWidth >> level, m_uHeight >> level), is = make_float2(1) / s;
		float2 l = a_UV * s - make_float2(0.5f);
		float ds = frac(l.x), dt = frac(l.y);
		return (1.f-ds) * (1.f-dt) * Texel<T, F>(func, level, a_UV) +
			   (1.f-ds) * dt       * Texel<T, F>(func, level, a_UV + make_float2(0, is.y)) +
			   ds       * (1.f-dt) * Texel<T, F>(func, level, a_UV + make_float2(is.x, 0)) +
			   ds       * dt       * Texel<T, F>(func, level, a_UV + make_float2(is.x, is.y));
	}
};

class e_MIPMap
{
	void* m_pDeviceData;
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
	e_MIPMap() {m_pDeviceData = 0; m_uWidth = m_uHeight = m_uBpp = -1;}
	e_MIPMap(float4& col);
	e_MIPMap(InputStream& a_In);
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