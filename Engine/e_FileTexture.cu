#include "StdAfx.h"
#include "e_FileTexture.h"
#include "e_ErrorHandler.h"
#define FREEIMAGE_LIB
#include <FreeImage.h>

Spectrum e_KernelMIPMap::Texel(unsigned int level, const float2& a_UV) const
{
	float2 l;
	if(!WrapCoordinates(a_UV, make_float2(m_uWidth >> level, m_uHeight >> level), m_uWrapMode, &l))
		return Spectrum(0.0f);
	else
	{
		unsigned int x = (unsigned int)l.x, y = (unsigned int)l.y;
		void* data;
#ifdef ISCUDA
		data = m_pDeviceData + (m_sOffsets[level] + y * (m_uWidth >> level) + x);
#else
		data = m_pHostData + (m_sOffsets[level] + y * (m_uWidth >> level) + x);
#endif
		Spectrum s;
		if(m_uType == vtRGBE)
			s.fromRGBE(*(RGBE*)data);
		else s.fromRGBCOL(*(RGBCOL*)data);
		return s;
	}
}

Spectrum e_KernelMIPMap::triangle(unsigned int level, const float2& a_UV) const
{
	level = clamp(level, 0u, m_uLevels-1);
	float2 s = make_float2(m_uWidth >> level, m_uHeight >> level), is = make_float2(1) / s;
	float2 l = a_UV * s;// - make_float2(0.5f)
	float ds = frac(l.x), dt = frac(l.y);
	return (1.f-ds) * (1.f-dt) * Texel(level, a_UV) +
			(1.f-ds) * dt       * Texel(level, a_UV + make_float2(0, is.y)) +
			ds       * (1.f-dt) * Texel(level, a_UV + make_float2(is.x, 0)) +
			ds       * dt       * Texel(level, a_UV + make_float2(is.x, is.y));
}

struct imgData
{
	unsigned int w, h;
	void* data;
	e_KernelTexture_DataType type;
};

bool parseImage(const char* a_InputFile, imgData* data)
{
	FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(a_InputFile, 0);
	if(fif == FIF_UNKNOWN)
	{
		fif = FreeImage_GetFIFFromFilename(a_InputFile);
	}
	if((fif != FIF_UNKNOWN) && FreeImage_FIFSupportsReading(fif))
	{
		FIBITMAP *dib = FreeImage_Load(fif, a_InputFile, 0);
		if(!dib)
			return false;
		unsigned int w = FreeImage_GetWidth(dib);
		unsigned int h = FreeImage_GetHeight(dib);
		unsigned int scan_width = FreeImage_GetPitch(dib);
		unsigned int pitch = FreeImage_GetPitch(dib);
		FREE_IMAGE_TYPE imageType = FreeImage_GetImageType(dib);
		unsigned int bpp = FreeImage_GetBPP(dib);
		BYTE *bits = (BYTE *)FreeImage_GetBits(dib);
		e_KernelTexture_DataType type = e_KernelTexture_DataType::vtRGBCOL;

		RGBCOL* tar = new RGBCOL[w * h], *ori = tar;
		if (((imageType == FIT_RGBAF) && (bpp == 128)) || ((imageType == FIT_RGBF) && (bpp == 96)))
		{
			type = e_KernelTexture_DataType::vtRGBE;
				for (unsigned int y = 0; y < h; ++y)
				{
						FIRGBAF *pixel = (FIRGBAF *)bits;
						for (unsigned int x = 0; x < w; ++x)
						{
							//*tar++ = Float4ToCOLORREF(make_float4(pixel->red, pixel->green, pixel->blue, pixel->alpha));
							*(RGBE*)tar++ = SpectrumConverter::Float3ToRGBE(make_float3(pixel->red, pixel->green, pixel->blue));
							pixel = (FIRGBAF*)((long long)pixel + bpp / 8);
						}
						bits += pitch;
				}
		}
		else if ((imageType == FIT_BITMAP) && ((bpp == 32) || (bpp == 24)))
		{

				for (unsigned int y = 0; y < h; ++y)
				{
						BYTE *pixel = (BYTE *)bits;
						for (unsigned int x = 0; x < w; ++x)
						{
							BYTE r = pixel[FI_RGBA_RED], g = pixel[FI_RGBA_GREEN], b = pixel[FI_RGBA_BLUE], a = bpp == 32 ? pixel[FI_RGBA_ALPHA] : 255;
							*tar++ = make_uchar4(r, g, b, a);
							pixel += bpp / 8;
						}
						bits += pitch;
				}
		}
		else if (bpp == 8)
		{
				for (unsigned int y = 0; y < h; ++y)
				{
						BYTE pixel;
						for (unsigned int x = 0; x < w; ++x)
						{
							FreeImage_GetPixelIndex(dib, x, y, &pixel);
							*tar++ = make_uchar4(pixel, pixel, pixel, 255);
						}
						bits += pitch;
				}
		}
		FreeImage_Unload(dib);

		data->type = type;
		data->h = h;
		data->w = w;
		data->data = ori;
		return true;
	}
	else
	{
		return false;
	}
}

Spectrum e_KernelMIPMap::Sample(const float2& uv) const
{
	return triangle(0, uv);
}

float e_KernelMIPMap::SampleAlpha(const float2& uv) const
{
	float2 l;
	if(!WrapCoordinates(uv, make_float2(m_uWidth, m_uHeight), m_uWrapMode, &l))
		return 0.0f;
	unsigned int x = (unsigned int)l.x, y = (unsigned int)l.y, level = 0;
	void* data;
#ifdef ISCUDA
			data = m_pDeviceData + (m_sOffsets[level] + y * (m_uWidth >> level) + x);
#else
			data = m_pHostData + (m_sOffsets[level] + y * (m_uWidth >> level) + x);
#endif
	if(m_uType == vtRGBE)
		return 1.0f;
	else return float(((RGBCOL*)data)->w) / 255.0f;
}

Spectrum e_KernelMIPMap::Sample(const float2& a_UV, float width) const
{
	float level = m_uLevels - 1 + Log2(MAX((float)width, 1e-8f));
	if (level < 0)
		return triangle(0, a_UV);
	else if (level >= m_uLevels - 1)
		return Texel(m_uLevels - 1, a_UV);
	else
	{
		int iLevel = Floor2Int(level);
		float delta = level - iLevel;
		return (1.f-delta) * triangle(iLevel, a_UV) + delta * triangle(iLevel+1, a_UV);
	}
}

Spectrum e_KernelMIPMap::Sample(float width, int x, int y) const
{
	float l = m_uLevels - 1 + Log2(MAX((float)width, 1e-8f));
	int level = (int)clamp(l, 0.0f, float(m_uLevels - 1));
	void* data;
#ifdef ISCUDA
		data = m_pDeviceData + (m_sOffsets[level] + y * (m_uWidth >> level) + x);
#else
		data = m_pHostData + (m_sOffsets[level] + y * (m_uWidth >> level) + x);
#endif
	Spectrum s;
	if(m_uType == vtRGBE)
		s.fromRGBE(*(RGBE*)data);
	else s.fromRGBCOL(*(RGBCOL*)data);
	return s;	
}

e_MIPMap::e_MIPMap(InputStream& a_In)
{
	a_In >> m_uWidth;
	a_In >> m_uHeight;
	a_In >> m_uBpp;
	a_In.operator>>(*(int*)&m_uType);
	a_In.operator>>(*(int*)&m_uWrapMode);
	a_In >> m_uLevels;
	a_In >> m_uSize;
	if(cudaMalloc(&m_pDeviceData, m_uSize))
		BAD_CUDA_ALLOC(m_uSize)
	m_pHostData = (unsigned int*)malloc(m_uSize);
	a_In.Read(m_pHostData, m_uSize);
	if(cudaMemcpy(m_pDeviceData, m_pHostData, m_uSize, cudaMemcpyHostToDevice))
			BAD_HOST_DEVICE_COPY(m_pDeviceData, m_uSize)
	a_In.Read(m_sOffsets, sizeof(m_sOffsets));
}

struct sampleHelper
{
	imgData* data;
	void* source, *dest;
	int w;

	sampleHelper(imgData* d, void* tmp, int i, int _w)
	{
		w = _w;
		data = d;
		source = i % 2 == 1 ? d->data : tmp;
		dest = i % 2 == 1 ? tmp : d->data;
	}

	float4 operator()(int x, int y)
	{
		if(data->type == vtRGBE)
			return make_float4(SpectrumConverter::RGBEToFloat3(((RGBE*)source)[y * w + x]));
		else return SpectrumConverter::COLORREFToFloat4(((RGBCOL*)source)[y * w + x]);
	}
};

void resize(imgData* d)
{
	int w = RoundUpPow2(d->w), h = RoundUpPow2(d->h);
	int* data = (int*)malloc(w * h * 4);
	for(int x = 0; x < w; x++)
		for(int y = 0; y < h; y++)
		{
			float x2 = float(d->w) * float(x) / float(w), y2 = float(d->h) * float(y) / float(h);
			data[y * w + x] = ((int*)d->data)[int(y2) * d->w + int(x2)];
		}
	free(d->data);
	d->data = data;
	d->w = w;
	d->h = h;
}

void e_MIPMap::CompileToBinary(const char* a_InputFile, OutputStream& a_Out, bool a_MipMap)
{
	imgData data;
	if(!parseImage(a_InputFile, &data))
		throw 1;
	if(popc(data.w) != 1 || popc(data.h) != 1)
		resize(&data);

	unsigned int nLevels = 1 + Log2Int(MAX(float(data.w), float(data.h)));
	if(!a_MipMap)
		nLevels = 1;
	unsigned int size = 0;
	for(unsigned int i = 0, j = data.w, k = data.h; i < nLevels; i++, j =  j >> 1, k = k >> 1)
		size += j * k * 4;
	void* buf = malloc(data.w * data.h * 4);//it will be smaller

	a_Out << data.w;
	a_Out << data.h;
	a_Out << unsigned int(4);
	a_Out << (int)data.type;
	a_Out << (int)TEXTURE_REPEAT;
	a_Out << nLevels;
	a_Out << size;
	a_Out.Write(data.data, data.w * data.h * sizeof(RGBCOL));

	unsigned int m_sOffsets[MAX_MIPS];
	m_sOffsets[0] = 0;
	unsigned int off = data.w * data.h;
	for(unsigned int i = 1, j = data.w / 2, k = data.h / 2; i < nLevels; i++, j >>= 1, k >>= 1)
	{
		sampleHelper H(&data, buf, i, j * 2);//last width
		for(unsigned int t = 0; t < k; t++)
			for(unsigned int s = 0; s < j; s++)
			{
				void* tar = (RGBE*)H.dest + t * j + s;
				float4 v = 0.25f * (H(2*s, 2*t) + H(2*s+1, 2*t) + H(2*s, 2*t+1) + H(2*s+1, 2*t+1));
				if(data.type == vtRGBE)
					*(RGBE*)tar = SpectrumConverter::Float3ToRGBE(make_float3(v));
				else *(RGBCOL*)tar = SpectrumConverter::Float4ToCOLORREF(v);
			}
		m_sOffsets[i] = off;
		off += j * k;
		a_Out.Write(H.dest, j * k * sizeof(RGBCOL));
	}
	a_Out.Write(m_sOffsets, sizeof(m_sOffsets));
	delete [] data.data;
}

e_KernelMIPMap e_MIPMap::CreateKernelTexture()
{
	e_KernelMIPMap r;
	r.m_pDeviceData = m_pDeviceData;
	r.m_pHostData = m_pHostData;
	r.m_uType = m_uType;
	r.m_uWrapMode = m_uWrapMode;
	r.m_uWidth = m_uWidth;
	r.m_uHeight = m_uHeight;
	r.m_uHeight = m_uHeight;
	for(int i = 0; i < MAX_MIPS; i++)
		r.m_sOffsets[i] = m_sOffsets[i];
	r.m_uLevels = m_uLevels;
	m_sKernelData = r;
	return r;
}