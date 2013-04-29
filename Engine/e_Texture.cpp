#include "StdAfx.h"
#include "e_Texture.h"
#include "e_ErrorHandler.h"
#include <FreeImage.h>

static char g_CopyData[4096 * 4096 * 16];

e_Texture::e_Texture(InputStream& a_In)
{
	a_In >> m_uWidth;
	a_In >> m_uHeight;
	a_In >> m_uBpp;
	unsigned int q = m_uWidth * m_uHeight * m_uBpp;
	if(cudaMalloc(&m_pDeviceData, q))
		BAD_CUDA_ALLOC(q)
	a_In.Read(g_CopyData, q);
	if(cudaMemcpy(m_pDeviceData, g_CopyData, q, cudaMemcpyHostToDevice))
			BAD_HOST_DEVICE_COPY(m_pDeviceData, q)
}

e_Texture::e_Texture(float4& col)
{
	m_uWidth = m_uHeight = 1;
	m_uBpp = 4;
	cudaMalloc(&m_pDeviceData, sizeof(RGBCOL));
	*(RGBCOL*)g_CopyData = Float4ToCOLORREF(col);
	if(cudaMemcpy(m_pDeviceData, g_CopyData, sizeof(RGBCOL), cudaMemcpyHostToDevice))
		BAD_HOST_DEVICE_COPY(m_pDeviceData, sizeof(RGBCOL))
}

void e_Texture::CompileToBinary(const char* a_InputFile, OutputStream& a_Out)
{
	FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(a_InputFile, 0);
	if(fif == FIF_UNKNOWN)
	{
		fif = FreeImage_GetFIFFromFilename(a_InputFile);
	}
	if((fif != FIF_UNKNOWN) && FreeImage_FIFSupportsReading(fif))
	{
		FIBITMAP *dib = FreeImage_Load(fif, a_InputFile, 0);
		
		unsigned int w = FreeImage_GetWidth(dib);
		unsigned int h = FreeImage_GetHeight(dib);
		unsigned int scan_width = FreeImage_GetPitch(dib);
		unsigned int pitch = FreeImage_GetPitch(dib);
		FREE_IMAGE_TYPE imageType = FreeImage_GetImageType(dib);
		unsigned int bpp = FreeImage_GetBPP(dib);
		BYTE *bits = (BYTE *)FreeImage_GetBits(dib);

		RGBCOL* tar = new RGBCOL[w * h], *ori = tar;
		if (((imageType == FIT_RGBAF) && (bpp == 128)) || ((imageType == FIT_RGBF) && (bpp == 96)))
		{
				for (unsigned int y = 0; y < h; ++y)
				{
						FIRGBAF *pixel = (FIRGBAF *)bits;
						for (unsigned int x = 0; x < w; ++x)
						{
							*tar++ = Float4ToCOLORREF(make_float4(pixel->red, pixel->green, pixel->blue, pixel->alpha));
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

		a_Out << w;
		a_Out << h;
		a_Out << unsigned int(4);
		a_Out.Write(ori, w * h * sizeof(RGBCOL));
	}
	else
	{
		throw 1;
	}
}

e_KernelTexture e_Texture::CreateKernelTexture()
{/*
	cudaResourceDesc            resDescr;
    memset(&resDescr,0,sizeof(cudaResourceDesc));

	resDescr.resType            = cudaResourceTypePitch2D;
	resDescr.res.pitch2D.desc = cudaCreateChannelDesc<RGBCOL>();
	resDescr.res.pitch2D.devPtr = m_pDeviceData;
	resDescr.res.pitch2D.height = m_uHeight;
	resDescr.res.pitch2D.width = m_uWidth;
	resDescr.res.pitch2D.pitchInBytes = m_uPitch;

    cudaTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = 1;
    texDescr.filterMode       = cudaFilterModeLinear;

    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
	texDescr.addressMode[2] = cudaAddressModeWrap;
	texDescr.readMode = cudaReadModeNormalizedFloat;

	cudaTextureObject_t obj;
    cudaError_t r = cudaCreateTextureObject(&obj, &resDescr, &texDescr, NULL);
	if(r)
		BAD_EXCEPTION("Bindless texture creation error : %s", cudaGetErrorString(r))
	m_sKernelData =  e_KernelTexture(obj);
	return m_sKernelData;*/
	e_KernelTexture r;
	r.m_fDim = make_float2(m_uWidth - 1, m_uHeight - 1);
	r.m_pDeviceData = m_pDeviceData;
	r.m_uWidth = m_uWidth;
	m_sKernelData = r;
	return r;
}