#include "StdAfx.h"
#include "e_Image.h"
#define FREEIMAGE_LIB
#include <FreeImage.h>
#include <stdexcept>
#include <iostream>


	//filter = filt;
	//filter.SetData(e_KernelGaussianFilter(2, 2, 0.55f));
	//filter.SetData(e_KernelMitchellFilter(1.0f/3.0f,1.0f/3.0, 4, 4));
	//filter.SetData(e_KernelLanczosSincFilter(4,4,5));
	//rebuildFilterTable();

e_Image::e_Image(int xRes, int yRes, unsigned int viewGLTexture)
	: xResolution(xRes), yResolution(yRes)
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
	: xResolution(xRes), yResolution(yRes)
{
ownsTarget = false;
	drawStyle = ImageDrawType::Normal;
	setStdFilter();
	CUDA_MALLOC(&cudaPixels, sizeof(Pixel) * xResolution * yResolution);
	hostPixels = new Pixel[xResolution * yResolution];
	outState = 1;
	isMapped = 0;
	cudaError er = cudaGraphicsD3D11RegisterResource(&viewCudaResource, pD3DResource, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	if(er)
		throw std::runtime_error(cudaGetErrorString(er));
	//use this to clear the array
	CUDA_MALLOC(&viewTarget, sizeof(RGBCOL) * xRes * yRes);
	ownsTarget = true;
	cudaMemset(viewTarget, 0, sizeof(RGBCOL) * xRes * yRes);
}
#endif

e_Image::e_Image(int xRes, int yRes, RGBCOL* target)
	: xResolution(xRes), yResolution(yRes)
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

void e_Image::rebuildFilterTable()
{
	float w = filter.As<e_KernelFilterBase>()->xWidth, h = filter.As<e_KernelFilterBase>()->yWidth;
	for (int y = 0; y < FILTER_TABLE_SIZE; ++y)
		for (int x = 0; x < FILTER_TABLE_SIZE; ++x)
		{
			float a = float(x) / FILTER_TABLE_SIZE * 2.0f - 1.0f, b = float(y) / FILTER_TABLE_SIZE * 2.0f - 1.0f;
			float _x = x + 0.5f, _y = y + 0.5f, s = FILTER_TABLE_SIZE;
//			filterTable[x][y] = filter.Evaluate(_x / s * w, _y / s * h);
			filterTable[x][y] = filter.Evaluate(a * w/2.0f, b * h/2.0f);
		}
	/*
	float* ftp = filterTable[0];
	for (int y = 0; y < FILTER_TABLE_SIZE; ++y)
	{
		float fy = ((float)y + .5f) * h / FILTER_TABLE_SIZE;
		for (int x = 0; x < FILTER_TABLE_SIZE; ++x)
		{
			float fx = ((float)x + .5f) * w / FILTER_TABLE_SIZE;
			*ftp++ = filter.Evaluate(fx, fy);
		}
	}*/
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
	uchar3* A = (uchar3*)FreeImage_GetBits(bitmap);
	for(int i = 0; i < xResolution * yResolution; i++)
	{
		A[i] = make_uchar3(colData[i].z, colData[i].y, colData[i].x);
	}
	delete [] colData;
	FreeImage_FlipVertical(bitmap);
	bool b = FreeImage_Save(ff, bitmap, fileName);
	FreeImage_Unload(bitmap);
}

void e_Image::WriteImage(const char* fileName, float splatScale)
{
	cudaMemcpy(hostPixels, cudaPixels, xResolution * yResolution * sizeof(Pixel), cudaMemcpyDeviceToHost);
	FIBITMAP* bitmap = FreeImage_Allocate(xResolution, yResolution, 96);
	BYTE *bits = (BYTE *)FreeImage_GetBits(bitmap);
    for (int y = 0; y < yResolution; ++y)
	{
		FIRGBAF *pixel = (FIRGBAF *)bits;
        for (int x = 0; x < xResolution; ++x)
		{
			hostPixels[y * xResolution + x].toSpectrum(splatScale).toLinearRGB(pixel->red, pixel->green, pixel->blue);
			pixel = (FIRGBAF*)((long long)pixel + 12);
		}
		bits += FreeImage_GetPitch(bitmap);
	}
	FreeImage_Save(FIF_EXR, bitmap, fileName);
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

//urgs
float gsplat;
void e_Image::DoUpdateDisplay(float splat)
{
	m_bDoUpdate = true;
	gsplat = splat;
}

void e_Image::EndRendering()
{
	if(m_bDoUpdate)
		InternalUpdateDisplay(gsplat);
	m_bDoUpdate = false;
	if(outState == 1)
	{
		cudaError r = cudaDestroySurfaceObject(viewCudaSurfaceObject);
		r = cudaGraphicsUnmapResources(1, &viewCudaResource);
	}
	isMapped = 0;
}