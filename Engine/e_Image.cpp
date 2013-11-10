#include "StdAfx.h"
#include "e_Image.h"
#define FREEIMAGE_LIB
#include <FreeImage.h>

e_Image::e_Image(const e_KernelFilter &filt, int xRes, int yRes, unsigned int viewGLTexture)
	: xResolution(xRes), yResolution(yRes)
{
	ownsTarget = false;
	doHDR = false;
	float crop[4] = {0, 1, 0, 1};
	filter = filt;
	//filter.SetData(e_KernelGaussianFilter(2, 2, 0.55f));
	//filter.SetData(e_KernelMitchellFilter(1.0f/3.0f,1.0f/3.0, 4, 4));
	//filter.SetData(e_KernelLanczosSincFilter(4,4,5));
	rebuildFilterTable();
	cudaMalloc(&cudaPixels, sizeof(Pixel) * xResolution * yResolution);
	hostPixels = new Pixel[xResolution * yResolution];
	outState = 1;
	isMapped = 0;
	this->viewGLTexture = viewGLTexture;
	cudaError er = cudaGraphicsGLRegisterImage(&viewCudaResource, viewGLTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	if(er)
		throw 1;
	//use this to clear the array
	cudaMalloc(&viewTarget, sizeof(RGBCOL) * xRes * yRes);
	ownsTarget = true;
	cudaMemset(viewTarget, 0, sizeof(RGBCOL) * xRes * yRes);
}

e_Image::e_Image(const e_KernelFilter &filt, int xRes, int yRes, RGBCOL* target)
	: xResolution(xRes), yResolution(yRes)
{
	doHDR = false;
	float crop[4] = {0, 1, 0, 1};
	filter = filt;
	//filter.SetData(e_KernelGaussianFilter(2, 2, 0.55f));
	//filter.SetData(e_KernelMitchellFilter(1.0f/3.0f,1.0f/3.0, 4, 4));
	//filter.SetData(e_KernelLanczosSincFilter(4,4,5));
	rebuildFilterTable();
	cudaMalloc(&cudaPixels, sizeof(Pixel) * xResolution * yResolution);
	hostPixels = new Pixel[xResolution * yResolution];
	outState = 2;
	isMapped = 0;
	this->viewTarget = target;
	if(!viewTarget)
	{
		cudaMalloc(&viewTarget, sizeof(RGBCOL) * xRes * yRes);
		ownsTarget = true;
	}
}

void e_Image::Free()
{
	delete hostPixels;
	cudaFree(cudaPixels);
	if(ownsTarget)
		cudaFree(viewTarget);
	if(outState == 1)
		cudaGraphicsUnregisterResource(viewCudaResource);
}

void e_Image::WriteDisplayImage(const char* fileName)
{
	FIBITMAP* bitmap = FreeImage_Allocate(xResolution, yResolution, 32, 0x000000ff, 0x0000ff00, 0x00ff0000);
	if(outState == 1)
	{
		cudaGraphicsMapResources(1, &viewCudaResource);
		cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, viewCudaResource, 0, 0);
		cudaMemcpyFromArray(FreeImage_GetBits(bitmap), viewCudaArray, 0, 0, xResolution * yResolution * sizeof(RGBCOL), cudaMemcpyDeviceToHost);
		cudaGraphicsUnmapResources(1, &viewCudaResource);
	}
	else
	{
		cudaMemcpy(FreeImage_GetBits(bitmap), viewTarget, sizeof(RGBCOL) * xResolution * yResolution, cudaMemcpyDeviceToHost);
	}
	RGBCOL* A = (RGBCOL*)FreeImage_GetBits(bitmap);
	for(int i = 0; i < xResolution * yResolution; i++)
	{
		unsigned char c = A[i].x;
		A[i].x = A[i].z;
		A[i].z = c;
	}
	bool b = FreeImage_Save(FIF_BMP, bitmap, fileName, BMP_DEFAULT);
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
	cudaError r = cudaDestroySurfaceObject(viewCudaSurfaceObject);
	r = cudaGraphicsUnmapResources(1, &viewCudaResource);
	isMapped = 0;
}