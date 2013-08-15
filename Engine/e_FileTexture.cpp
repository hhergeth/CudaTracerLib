#include "StdAfx.h"
#include "e_FileTexture.h"
#include "e_ErrorHandler.h"
#include <FreeImage.h>

static char g_CopyData[4096 * 4096 * 16];

/*
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

struct imgData
{
	unsigned int w, h, b;
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
			RGBE* tar2 = (RGBE*)tar;
				for (unsigned int y = 0; y < h; ++y)
				{
						FIRGBAF *pixel = (FIRGBAF *)bits;
						for (unsigned int x = 0; x < w; ++x)
						{
							//*tar++ = Float4ToCOLORREF(make_float4(pixel->red, pixel->green, pixel->blue, pixel->alpha));
							*(RGBE*)tar++ = Float3ToRGBE(make_float3(pixel->red, pixel->green, pixel->blue));
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
		data->b = 4;
		data->h = h;
		data->w = w;
		data->data = ori;
	}
	else
	{
		return false;
	}
}

e_FileTexture::e_FileTexture(InputStream& a_In)
{
	a_In >> m_uWidth;
	a_In >> m_uHeight;
	a_In >> m_uBpp;
	a_In.operator>>(*(int*)&m_uType);
	a_In.operator>>(*(int*)&m_uWrapMode);
	unsigned int q = m_uWidth * m_uHeight * m_uBpp;
	if(cudaMalloc(&m_pDeviceData, q))
		BAD_CUDA_ALLOC(q)
	a_In.Read(g_CopyData, q);
	if(cudaMemcpy(m_pDeviceData, g_CopyData, q, cudaMemcpyHostToDevice))
			BAD_HOST_DEVICE_COPY(m_pDeviceData, q)
}

e_FileTexture::e_FileTexture(float4& col)
{
	m_uWrapMode = TEXTURE_REPEAT;
	m_uWidth = m_uHeight = 1;
	m_uBpp = 4;
	cudaMalloc(&m_pDeviceData, sizeof(RGBCOL));
	*(RGBCOL*)g_CopyData = Float4ToCOLORREF(col);
	if(cudaMemcpy(m_pDeviceData, g_CopyData, sizeof(RGBCOL), cudaMemcpyHostToDevice))
		BAD_HOST_DEVICE_COPY(m_pDeviceData, sizeof(RGBCOL))
	m_uType = e_KernelTexture_DataType::vtGeneric;
}

void e_FileTexture::CompileToBinary(const char* a_InputFile, OutputStream& a_Out)
{
	imgData data;
	if(!parseImage(a_InputFile, &data))
		throw 1;
	else
	{
		a_Out << data.w;
		a_Out << data.h;
		a_Out << unsigned int(4);
		a_Out << (int)data.type;
		a_Out << (int)TEXTURE_REPEAT;
		a_Out.Write(data.data, data.w * data.h * sizeof(RGBCOL));
		delete [] data.data;
	}
}

e_KernelFileTexture e_FileTexture::CreateKernelTexture()
{
	e_KernelFileTexture r;
	r.m_fDim = make_float2(m_uWidth - 1, m_uHeight - 1);
	r.m_pDeviceData = m_pDeviceData;
	r.m_uWidth = m_uWidth;
	r.m_uType = m_uType;
	r.m_uDim = make_int2(m_uWidth-1, m_uHeight-1);
	r.m_uWrapMode = m_uWrapMode;
	m_sKernelData = r;
	return r;
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
	a_In.Read(g_CopyData, m_uSize);
	if(cudaMemcpy(m_pDeviceData, g_CopyData, m_uSize, cudaMemcpyHostToDevice))
			BAD_HOST_DEVICE_COPY(m_pDeviceData, m_uSize)
	a_In.Read(m_sOffsets, sizeof(m_sOffsets));
}

e_MIPMap::e_MIPMap(float4& col)
{
	m_uLevels = 1;
	m_uSize = 4;
	m_uWrapMode = TEXTURE_REPEAT;
	m_uWidth = m_uHeight = 1;
	m_uBpp = 4;
	cudaMalloc(&m_pDeviceData, sizeof(RGBCOL));
	*(RGBCOL*)g_CopyData = Float4ToCOLORREF(col);
	if(cudaMemcpy(m_pDeviceData, g_CopyData, sizeof(RGBCOL), cudaMemcpyHostToDevice))
		BAD_HOST_DEVICE_COPY(m_pDeviceData, sizeof(RGBCOL))
	m_uType = e_KernelTexture_DataType::vtRGBCOL;
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
			return make_float4(RGBEToFloat3(((RGBE*)source)[y * w + x]));
		else return COLORREFToFloat4(((RGBCOL*)source)[y * w + x]);
	}
};

void e_MIPMap::CompileToBinary(const char* a_InputFile, OutputStream& a_Out)
{
	imgData data;
	if(!parseImage(a_InputFile, &data))
		throw 1;
	else if(__popc(data.w) == 1)//2^i
	{
		unsigned int nLevels = 1 + Log2Int(float(MAX(data.w, data.h)));
		unsigned int size = 0;
		for(int i = 0, j = data.w, k = data.h; i < nLevels; i++, j =  j >> 1, k = k >> 1)
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
		for(int i = 1, j = data.w / 2, k = data.h / 2; i < nLevels; i++, j =  j >> 1, k = k >> 1)
		{
			sampleHelper H(&data, buf, i, j * 2);//last width
			for(int t = 0; t < k; t++)
				for(int s = 0; s < j; s++)
				{
					void* tar = (RGBE*)H.dest + t * j + s;
					float4 v = 0.25f * (H(2*s, 2*t) + H(2*s+1, 2*t) + H(2*s, 2*t+1) + H(2*s+1, 2*t+1));
					if(data.type == vtRGBE)
						*(RGBE*)tar = Float3ToRGBE(make_float3(v));
					else *(RGBCOL*)tar = Float4ToCOLORREF(v);
				}
			m_sOffsets[i] = off;
			off += j * k;
			a_Out.Write(H.dest, j * k * sizeof(RGBCOL));
		}
		a_Out.Write(m_sOffsets, sizeof(m_sOffsets));
		delete [] data.data;
	}
	else throw 1;
}

e_KernelMIPMap e_MIPMap::CreateKernelTexture()
{
	e_KernelMIPMap r;
	r.m_pDeviceData = m_pDeviceData;
	r.m_uType = m_uType;
	r.m_uWrapMode = m_uWrapMode;
	r.m_uWidth = m_uWidth;
	r.m_uHeight = m_uHeight;
	for(int i = 0; i < MAX_MIPS; i++)
		r.m_sOffsets[i] = m_sOffsets[i];
	r.m_uLevels = m_uLevels;
	m_sKernelData = r;
	return r;
}