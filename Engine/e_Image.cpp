#include "StdAfx.h"
#include "e_Image.h"
#include <FreeImage.h>

e_Image::e_Image(e_KernelFilter &filt, const float crop[4], int xRes, int yRes, RGBCOL* cudaBuffer)
	: xResolution(xRes), yResolution(yRes)
{
	filter = filt;
    memcpy(cropWindow, crop, 4 * sizeof(float));
	xPixelStart = Ceil2Int(xResolution * cropWindow[0]);
    xPixelCount = MAX(1, Ceil2Int(xResolution * cropWindow[1]) - xPixelStart);
    yPixelStart = Ceil2Int(yResolution * cropWindow[2]);
    yPixelCount = MAX(1, Ceil2Int(yResolution * cropWindow[3]) - yPixelStart);
	float *ftp = filterTable;
	float w = filter.As<e_KernelFilterBase>()->xWidth, h = filter.As<e_KernelFilterBase>()->yWidth;
    for (int y = 0; y < FILTER_TABLE_SIZE; ++y)
	{
        float fy = ((float)y + .5f) * h / FILTER_TABLE_SIZE;
        for (int x = 0; x < FILTER_TABLE_SIZE; ++x)
		{
            float fx = ((float)x + .5f) * w / FILTER_TABLE_SIZE;
            *ftp++ = filter.Evaluate(fx, fy);
        }
    }
	target = cudaBuffer;
	cudaMalloc(&cudaPixels, sizeof(Pixel) * xPixelCount * yPixelCount);
	hostPixels = new Pixel[xPixelCount * yPixelCount];
}

e_Image::e_Image(e_KernelFilter &filt, int xRes, int yRes, RGBCOL* cudaBuffer)
	: xResolution(xRes), yResolution(yRes)
{
	float crop[4] = {0, 1, 0, 1};
	filter = filt;
	//filter.SetData(e_KernelGaussianFilter(4, 4, 0.55f));
	//filter.SetData(e_KernelMitchellFilter(1.0f/3.0f,1.0f/3.0, 4, 4));
	//filter.SetData(e_KernelLanczosSincFilter(4,4,5));
    memcpy(cropWindow, crop, 4 * sizeof(float));
	xPixelStart = Ceil2Int(xResolution * cropWindow[0]);
    xPixelCount = MAX(1, Ceil2Int(xResolution * cropWindow[1]) - xPixelStart);
    yPixelStart = Ceil2Int(yResolution * cropWindow[2]);
    yPixelCount = MAX(1, Ceil2Int(yResolution * cropWindow[3]) - yPixelStart);
	float *ftp = filterTable;
	float w = filter.As<e_KernelFilterBase>()->xWidth, h = filter.As<e_KernelFilterBase>()->yWidth;
    for (int y = 0; y < FILTER_TABLE_SIZE; ++y)
	{
        float fy = ((float)y + .5f) * h / FILTER_TABLE_SIZE;
        for (int x = 0; x < FILTER_TABLE_SIZE; ++x)
		{
            float fx = ((float)x + .5f) * w / FILTER_TABLE_SIZE;
            *ftp++ = filter.Evaluate(fx, fy);
        }
    }
	target = cudaBuffer;
	cudaMalloc(&cudaPixels, sizeof(Pixel) * xPixelCount * yPixelCount);
	hostPixels = new Pixel[xPixelCount * yPixelCount];
}

void e_Image::WriteDisplayImage(const char* fileName)
{
	FIBITMAP* bitmap = FreeImage_Allocate(xResolution, yResolution, 32, 0x000000ff, 0x0000ff00, 0x00ff0000);
	cudaMemcpy(FreeImage_GetBits(bitmap), target, sizeof(RGBCOL) * xResolution * yResolution, cudaMemcpyDeviceToHost);
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
			float weightSum = hostPixels[y * xResolution + x].weightSum;
			Spectrum rgb = hostPixels[y * xResolution + x].rgb;
			if(weightSum != 0)
				rgb = fmaxf(Spectrum(0.0f), rgb / hostPixels[y * xResolution + x].weightSum);
			rgb += splatScale * hostPixels[y * xResolution + x].splatRgb;
			rgb.toLinearRGB(pixel->red, pixel->green, pixel->blue);
			pixel = (FIRGBAF*)((long long)pixel + 12);
		}
		bits += FreeImage_GetPitch(bitmap);
	}
	FreeImage_Save(FIF_EXR, bitmap, fileName);
	FreeImage_Unload(bitmap);
}

void e_Image::StartNewRendering()
{
	usedHostPixels = false;
	ZeroMemory(hostPixels, sizeof(Pixel) * xResolution * yResolution);
	cudaMemset(cudaPixels, 0, sizeof(Pixel) * xResolution * yResolution);
	cudaMemset(target, 0, sizeof(RGBCOL) * xResolution * yResolution);
}