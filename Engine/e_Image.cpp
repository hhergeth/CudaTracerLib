#include "StdAfx.h"
#include "e_Image.h"
#define FREEIMAGE_LIB
#include <FreeImage.h>
#include <stdexcept>
#include <iostream>

e_Image::e_Image(int xRes, int yRes, unsigned int viewGLTexture)
	: xResolution(xRes), yResolution(yRes), lastSplatVal(0)
{
	ownsTarget = false;
	drawStyle = ImageDrawType::Normal;
	setStdFilter();
	CUDA_MALLOC(&cudaPixels, sizeof(Pixel) * xResolution * yResolution);
	hostPixels = new Pixel[xResolution * yResolution];
	outState = 1;
	isMapped = 0;
	cudaError er = cudaGraphicsGLRegisterImage(&viewCudaResource, viewGLTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	if(er)
		throw std::runtime_error(cudaGetErrorString(er));
	//use this to clear the array
	CUDA_MALLOC(&viewTarget, sizeof(RGBCOL) * xRes * yRes);
	ownsTarget = true;
	cudaMemset(viewTarget, 0, sizeof(RGBCOL) * xRes * yRes);
}

#ifdef ISWINDOWS
#include <cuda_d3d11_interop.h>
e_Image::e_Image(int xRes, int yRes, ID3D11Resource *pD3DResource)
	: xResolution(xRes), yResolution(yRes), lastSplatVal(0)
{
ownsTarget = false;
	drawStyle = ImageDrawType::Normal;
	setStdFilter();
	CUDA_MALLOC(&cudaPixels, sizeof(Pixel) * xResolution * yResolution);
	hostPixels = new Pixel[xResolution * yResolution];
	outState = 1;
	isMapped = 0;
	cudaError er = cudaGraphicsD3D11RegisterResource(&viewCudaResource, pD3DResource, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	if (er)
	{
		std::cout << cudaGetErrorString(er) << "\n";
		throw std::runtime_error(cudaGetErrorString(er));
	}
	//use this to clear the array
	CUDA_MALLOC(&viewTarget, sizeof(RGBCOL) * xRes * yRes);
	ownsTarget = true;
	cudaMemset(viewTarget, 0, sizeof(RGBCOL) * xRes * yRes);
}
#endif

e_Image::e_Image(int xRes, int yRes, RGBCOL* target)
	: xResolution(xRes), yResolution(yRes), lastSplatVal(0)
{
	drawStyle = ImageDrawType::Normal;
	setStdFilter();
	CUDA_MALLOC(&cudaPixels, sizeof(Pixel) * xResolution * yResolution);
	hostPixels = new Pixel[xResolution * yResolution];
	outState = 2;
	isMapped = 0;
	this->viewTarget = target;
	if(!viewTarget)
	{
		CUDA_MALLOC(&viewTarget, sizeof(RGBCOL) * xRes * yRes);
		ownsTarget = true;
	}
}

void e_Image::Free()
{
	delete hostPixels;
	CUDA_FREE(cudaPixels);
	if(ownsTarget)
		CUDA_FREE(viewTarget);
	if(outState == 1)
		cudaGraphicsUnregisterResource(viewCudaResource);
}

void e_Image::WriteDisplayImage(const char* fileName)
{
	FREE_IMAGE_FORMAT ff = FreeImage_GetFIFFromFilename(fileName);

	RGBCOL* colData = new RGBCOL[xResolution * yResolution];
	if(outState == 1)
	{
		cudaGraphicsMapResources(1, &viewCudaResource);
		cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, viewCudaResource, 0, 0);
		cudaMemcpyFromArray(colData, viewCudaArray, 0, 0, xResolution * yResolution * sizeof(RGBCOL), cudaMemcpyDeviceToHost);
		cudaGraphicsUnmapResources(1, &viewCudaResource);
	}
	else
	{
		cudaMemcpy(colData, viewTarget, sizeof(RGBCOL) * xResolution * yResolution, cudaMemcpyDeviceToHost);
	}
	FIBITMAP* bitmap = FreeImage_Allocate(xResolution, yResolution, 24, 0x000000ff, 0x0000ff00, 0x00ff0000);
	BYTE* A = FreeImage_GetBits(bitmap);
	unsigned int pitch = FreeImage_GetPitch(bitmap);
	int off = 0;
	for(int y = 0; y < yResolution; y++)
	{
		for(int x = 0; x < xResolution; x++)
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
	delete [] colData;
	bool b = FreeImage_Save(ff, bitmap, fileName);
	FreeImage_Unload(bitmap);
}

void e_Image::StartRendering()
{
	m_bDoUpdate = false;
	if(outState == 1)
	{
		cudaError r = cudaGraphicsMapResources(1, &viewCudaResource);
		r = cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, viewCudaResource, 0, 0);
		cudaResourceDesc viewCudaArrayResourceDesc;
		viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
		viewCudaArrayResourceDesc.res.array.array = viewCudaArray;
		r = cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc);
		isMapped = 1;
	}
}

void e_Image::DoUpdateDisplay(float splat)
{
	m_bDoUpdate = true;
	lastSplatVal = splat;
}

void e_Image::EndRendering()
{
	if(m_bDoUpdate)
		InternalUpdateDisplay();
	m_bDoUpdate = false;
	if(outState == 1)
	{
		cudaError r = cudaDestroySurfaceObject(viewCudaSurfaceObject);
		r = cudaGraphicsUnmapResources(1, &viewCudaResource);
	}
	isMapped = 0;
}