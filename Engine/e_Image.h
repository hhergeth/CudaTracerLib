#pragma once

#include "e_Filter.h"


#define FILTER_TABLE_SIZE 16

class e_Image
{
public:
    // ImageFilm Public Methods
	CUDA_FUNC_IN e_Image(){}
    e_Image(e_KernelFilter &filt, const float crop[4], int xRes, int yRes, RGBCOL* cudaBuffer);
	e_Image(e_KernelFilter &filt, int xRes, int yRes, RGBCOL* cudaBuffer);
    void Free()
	{
	    delete hostPixels;
		cudaFree(cudaPixels);
	}
#ifdef __CUDACC__
    CUDA_ONLY_FUNC void AddSample(const CameraSample &sample, const float3 &L)
	{/*
		float dimageX = sample.imageX - 0.5f;
		float dimageY = sample.imageY - 0.5f;
		int x0 = Ceil2Int (dimageX - filter.As<e_KernelFilterBase>()->xWidth);
		int x1 = Floor2Int(dimageX + filter.As<e_KernelFilterBase>()->xWidth);
		int y0 = Ceil2Int (dimageY - filter.As<e_KernelFilterBase>()->yWidth);
		int y1 = Floor2Int(dimageY + filter.As<e_KernelFilterBase>()->yWidth);
		x0 = MAX(x0, xPixelStart);
		x1 = MIN(x1, xPixelStart + xPixelCount - 1);
		y0 = MAX(y0, yPixelStart);
		y1 = MIN(y1, yPixelStart + yPixelCount - 1);
		if ((x1-x0) < 0 || (y1-y0) < 0)
			return;
		float invX = filter.As<e_KernelFilterBase>()->invXWidth, invY = filter.As<e_KernelFilterBase>()->invYWidth;
		for (int y = y0; y <= y1; ++y)
		{
			for (int x = x0; x <= x1; ++x)
			{
				// Evaluate filter value at $(x,y)$ pixel
				float fx = fabsf((x - dimageX) * invX * FILTER_TABLE_SIZE);
				float fy = fabsf((y - dimageY) * invY * FILTER_TABLE_SIZE);
				int ify = MIN(Floor2Int(fx), FILTER_TABLE_SIZE-1);
				int ifx = MIN(Floor2Int(fy), FILTER_TABLE_SIZE-1);
				int offset = ify * FILTER_TABLE_SIZE + ifx;
				float filterWt = filterTable[offset];

				// Update pixel values with filtered sample contribution
				Pixel* pixel = cudaPixels + ((y - yPixelStart) * xPixelCount + (x - xPixelStart));
				atomicAdd(&pixel->rgb.x, filterWt * L.x);
                atomicAdd(&pixel->rgb.y, filterWt * L.y);
                atomicAdd(&pixel->rgb.z, filterWt * L.z);
                atomicAdd(&pixel->weightSum, filterWt);
			}
		}*/
		int x = (int)sample.imageX, y = (int)sample.imageY;
		Pixel* pixel = cudaPixels + ((y - yPixelStart) * xPixelCount + (x - xPixelStart));
		pixel->rgb += L;
		pixel->weightSum += 1;
	}
    CUDA_ONLY_FUNC void Splat(const CameraSample &sample, const float3 &L)
	{
		int x = Floor2Int(sample.imageX), y = Floor2Int(sample.imageY);
		if (x < xPixelStart || x - xPixelStart >= xPixelCount || y < yPixelStart || y - yPixelStart >= yPixelCount)
			return;
		Pixel* pixel = cudaPixels + ((y - yPixelStart) * xPixelCount + (x - xPixelStart));
		atomicAdd(&pixel->splatRgb.x, L.x);
		atomicAdd(&pixel->splatRgb.y, L.y);
		atomicAdd(&pixel->splatRgb.z, L.z);
	}
	CUDA_ONLY_FUNC void SetSampleDirect(const CameraSample &sample, const float3 &L)
	{
		int x = Floor2Int(sample.imageX), y = Floor2Int(sample.imageY);
		if (x < xPixelStart || x - xPixelStart >= xPixelCount || y < yPixelStart || y - yPixelStart >= yPixelCount)
			return;
		unsigned int off = (y - yPixelStart) * xPixelCount + (x - xPixelStart);
		Pixel* pixel = cudaPixels + off;
		pixel->rgb = L;
		pixel->weightSum = 1;
		pixel->splatRgb = make_float3(0);
		target[off] = Float3ToCOLORREF(L);
	}
#endif
    CUDA_FUNC_IN void GetSampleExtent(int *xstart, int *xend, int *ystart, int *yend)
	{
		*xstart = Floor2Int(xPixelStart + 0.5f - filter.As<e_KernelFilterBase>()->xWidth);
		*xend   = Floor2Int(xPixelStart + 0.5f + xPixelCount  + filter.As<e_KernelFilterBase>()->xWidth);

		*ystart = Floor2Int(yPixelStart + 0.5f - filter.As<e_KernelFilterBase>()->yWidth);
		*yend   = Floor2Int(yPixelStart + 0.5f + yPixelCount + filter.As<e_KernelFilterBase>()->yWidth);
	}
    CUDA_FUNC_IN void GetPixelExtent(int *xstart, int *xend, int *ystart, int *yend)
	{
		    *xstart = xPixelStart;
		*xend   = xPixelStart + xPixelCount;
		*ystart = yPixelStart;
		*yend   = yPixelStart + yPixelCount;
	}
    void WriteDisplayImage(const char* fileName);
	void WriteImage(const char* fileName, float splatScale = 1.0f);
    void UpdateDisplay(float splatScale = 1.0f);
	void StartNewRendering();
	struct Pixel {
        Pixel() {
            rgb = splatRgb = make_float3(0);
            weightSum = 0.f;
        }
        float3 rgb;
        float weightSum;
        float3 splatRgb;
        RGBCOL pad;
    };
private:
    // ImageFilm Private Data
	e_KernelFilter filter;
    float cropWindow[4];
    int xPixelStart, yPixelStart, xPixelCount, yPixelCount;
    Pixel *cudaPixels;
	Pixel *hostPixels;
	RGBCOL* target;
	float filterTable[FILTER_TABLE_SIZE * FILTER_TABLE_SIZE];
	int xResolution, yResolution;
};