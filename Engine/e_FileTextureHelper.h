#pragma once

#define FREEIMAGE_LIB
#include <FreeImage.h>
#include "e_FileTexture.h"

struct imgData
{
	unsigned int w, h;
	void* data;
	e_KernelTexture_DataType type;

	CUDA_FUNC_IN Spectrum Load(int x, int y)
	{
		Spectrum s;
		if(type == vtRGBE)
			s.fromRGBE(((RGBE*)data)[y * w + x]);
		else s.fromRGBCOL(((RGBCOL*)data)[y * w + x]);
		return s;
	}
};

inline bool parseImage(const char* a_InputFile, imgData* data)
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