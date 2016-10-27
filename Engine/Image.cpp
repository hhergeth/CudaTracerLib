#include "StdAfx.h"
#include "Image.h"
#include <stdexcept>
#include <iostream>
#include <CudaMemoryManager.h>

#ifdef ISWINDOWS
#include <windows.h>
#include <cuda_d3d11_interop.h>
#endif
#include <cuda_gl_interop.h>

#define FREEIMAGE_LIB
#include <FreeImage.h>

namespace CudaTracerLib {

Image::Image(int xRes, int yRes, RGBCOL* target)
	: xResolution(xRes), yResolution(yRes), m_fOutScale(1), lastSplatVal(0)
{
	ThrowCudaErrors();
	setStdFilter();
	CUDA_MALLOC(&cudaPixels, sizeof(Pixel) * xResolution * yResolution);
	hostPixels = new Pixel[xResolution * yResolution];
	this->viewTarget = target;
	ownsTarget = false;
	if (!viewTarget)
	{
		CUDA_MALLOC(&viewTarget, sizeof(RGBCOL) * xRes * yRes);
		ownsTarget = true;
	}
	CUDA_MALLOC(&m_filteredColorsDevice, sizeof(RGBE) * xRes * yRes);
}

void Image::Free()
{
	ThrowCudaErrors();
	delete hostPixels;
	CUDA_FREE(cudaPixels);
	if (ownsTarget)
		CUDA_FREE(viewTarget);
	CUDA_FREE(m_filteredColorsDevice);
}

void Image::copyToHost()
{
	ThrowCudaErrors(cudaMemcpy(hostPixels, cudaPixels, sizeof(Pixel) * xResolution * yResolution, cudaMemcpyDeviceToHost));
}

FIBITMAP* Image::toFreeImage()
{
	static const int xDim = 4096;
	static const int yDim = 4096;
	static RGBCOL colData[xDim * yDim];
	if (yResolution > yDim || xResolution > xDim)
		throw std::runtime_error("Image resolution too high!");

	ThrowCudaErrors(cudaMemcpy(colData, viewTarget, sizeof(RGBCOL) * xResolution * yResolution, cudaMemcpyDeviceToHost));
	FIBITMAP* bitmap = FreeImage_Allocate(xResolution, yResolution, 24, 0x000000ff, 0x0000ff00, 0x00ff0000);
	BYTE* A = FreeImage_GetBits(bitmap);
	unsigned int pitch = FreeImage_GetPitch(bitmap);
	int off = 0;
	for (int y = 0; y < yResolution; y++)
	{
		for (int x = 0; x < xResolution; x++)
		{
			int i = (yResolution - 1 - y) * xResolution + x;
			Spectrum rgb = Spectrum(colData[i].z, colData[i].y, colData[i].x) / 255;
			//Spectrum srgb;
			//rgb.toSRGB(srgb[0], srgb[1], srgb[2]);
			//RGBCOL p = srgb.toRGBCOL();
			RGBCOL p = make_uchar4(colData[i].z, colData[i].y, colData[i].x, 255);
			//RGBCOL p = Spectrum(rgb.pow(1.0f / 2.2f)).toRGBCOL();
			A[off + x * 3 + 0] = p.x;
			A[off + x * 3 + 1] = p.y;
			A[off + x * 3 + 2] = p.z;
		}
		off += pitch;
	}
	return bitmap;
}

void Image::WriteDisplayImage(const std::string& fileName)
{
	FIBITMAP* bitmap = toFreeImage();
	FREE_IMAGE_FORMAT ff = FreeImage_GetFIFFromFilename(fileName.c_str());
	int flags = ff == FREE_IMAGE_FORMAT::FIF_JPEG ? JPEG_QUALITYSUPERB : 0;
	if (!FreeImage_Save(ff, bitmap, fileName.c_str(), flags))
		throw std::runtime_error("Failed saving Screenshot!");
	FreeImage_Unload(bitmap);
}

void Image::SaveToMemory(void** mem, size_t& size, const std::string& type)
{
	FIBITMAP* bitmap = toFreeImage();
	FREE_IMAGE_FORMAT ff = FreeImage_GetFIFFromFilename(type.c_str());
	FIMEMORY* str = FreeImage_OpenMemory();
	int flags = ff == FREE_IMAGE_FORMAT::FIF_JPEG ? JPEG_QUALITYSUPERB : 0;
	if (!FreeImage_SaveToMemory(ff, bitmap, str, flags))
		throw std::runtime_error("SaveToMemory::FreeImage_SaveToMemory");
	long file_size = FreeImage_TellMemory(str);
	if (*mem == 0 || file_size > size)
	{
		if (*mem)
			free(*mem);
		size = file_size;
		*mem = malloc(file_size);
	}
	FreeImage_SeekMemory(str, 0L, SEEK_SET);
	unsigned n = FreeImage_ReadMemory(*mem, 1, file_size, str);
	if (n != file_size)
		throw std::runtime_error("SaveToMemory::FreeImage_ReadMemory");
	FreeImage_CloseMemory(str);
	FreeImage_Unload(bitmap);
}

void Image::StartRendering()
{
	m_bDoUpdate = false;
}

void Image::DoUpdateDisplay(float splat)
{
	m_bDoUpdate = true;
	lastSplatVal = splat;
}

void Image::EndRendering()
{
	if (m_bDoUpdate)
		InternalUpdateDisplay();
	m_bDoUpdate = false;
}

}
