#pragma once

#include "e_Filter.h"
#include "e_Samples.h"

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
    CUDA_DEVICE CUDA_HOST void AddSample(const CameraSample &sample, const Spectrum &L);
    CUDA_DEVICE CUDA_HOST void Splat(const CameraSample &sample, const Spectrum &L);
	CUDA_DEVICE CUDA_HOST void SetSampleDirect(const CameraSample &sample, const Spectrum &L);
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
        Spectrum rgb;
        float weightSum;
        Spectrum splatRgb;
        RGBCOL pad;
    };
private:
    // ImageFilm Private Data
	e_KernelFilter filter;
    float cropWindow[4];
    int xPixelStart, yPixelStart, xPixelCount, yPixelCount;
    Pixel *cudaPixels;
	Pixel *hostPixels;
	bool usedHostPixels;
	RGBCOL* target;
	float filterTable[FILTER_TABLE_SIZE * FILTER_TABLE_SIZE];
	int xResolution, yResolution;
	CUDA_FUNC_IN Pixel* getPixel(int i)
	{
#ifdef __CUDACC__
		return cudaPixels + i;
#else
		usedHostPixels = true;
		return hostPixels + i;
#endif
	}
};