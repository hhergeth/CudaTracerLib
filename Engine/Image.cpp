#include "StdAfx.h"
#include "Image.h"
#include <stdexcept>
#include <iostream>
#include <Base/CudaMemoryManager.h>

#define FREEIMAGE_LIB
#include <FreeImage/FreeImage.h>

namespace CudaTracerLib {

Image::Image(int xRes, int yRes, RGBCOL* target)
	: xResolution(xRes), yResolution(yRes), ISynchronizedBufferParent(m_pixelBuffer), m_pixelBuffer(xRes * yRes)
{
	CUDA_MALLOC(&m_filteredColorsDevice, sizeof(RGBE) * xRes * yRes);
	m_viewTarget = target;
	ownsTarget = false;
	if (!m_viewTarget)
	{
		CUDA_MALLOC(&m_viewTarget, sizeof(RGBCOL) * xRes * yRes);
		ownsTarget = true;
	}
}

void Image::Free()
{
	CUDA_FREE(m_filteredColorsDevice);
	if (ownsTarget)
		CUDA_FREE(m_viewTarget);
}

FIBITMAP* Image::toFreeImage(bool HDR)
{
	RGBCOL* colData = new RGBCOL[xResolution * yResolution];

	ThrowCudaErrors(cudaMemcpy(colData, HDR ? m_filteredColorsDevice : m_viewTarget, (HDR ? sizeof(RGBE) : sizeof(RGBCOL)) * xResolution * yResolution, cudaMemcpyDeviceToHost));
	FIBITMAP* bitmap = HDR ? FreeImage_AllocateT(FIT_RGBF, xResolution, yResolution)
						   : FreeImage_Allocate(xResolution, yResolution, 24, 0x000000ff, 0x0000ff00, 0x00ff0000);
	BYTE* A = FreeImage_GetBits(bitmap);
	unsigned int pitch = FreeImage_GetPitch(bitmap);
	int off = 0;
	for (int y = 0; y < yResolution; y++)
	{
		for (int x = 0; x < xResolution; x++)
		{
			int i = (yResolution - 1 - y) * xResolution + x;
			if (HDR)
			{
				Vec3f* p = (Vec3f*)(A + off + x * sizeof(Vec3f));
				Spectrum s;
				s.fromRGBE(*((RGBE*)colData + i));
				s.toLinearRGB(p->x, p->y, p->z);
			}
			else
			{
				A[off + x * 3 + 0] = colData[i].z;
				A[off + x * 3 + 1] = colData[i].y;
				A[off + x * 3 + 2] = colData[i].x;
			}
		}
		off += pitch;
	}
	delete[] colData;
	return bitmap;
}

void Image::WriteDisplayImage(const std::string& fileName)
{
	FREE_IMAGE_FORMAT ff = FreeImage_GetFIFFromFilename(fileName.c_str());
	FIBITMAP* bitmap = toFreeImage(ff == FREE_IMAGE_FORMAT::FIF_HDR || ff == FREE_IMAGE_FORMAT::FIF_EXR);
	int flags = ff == FREE_IMAGE_FORMAT::FIF_JPEG ? JPEG_QUALITYSUPERB : 0;
	if (!FreeImage_Save(ff, bitmap, fileName.c_str(), flags))
		throw std::runtime_error("Failed saving Screenshot!");
	FreeImage_Unload(bitmap);
}

void Image::SaveToMemory(void** mem, size_t& size, const std::string& type)
{
	FIBITMAP* bitmap = toFreeImage(false);
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

}
