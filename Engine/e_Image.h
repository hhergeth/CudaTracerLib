#pragma once

#include "e_Filter.h"
#include "e_Samples.h"

#ifdef ISWINDOWS
#include <windows.h>
#include <cuda_gl_interop.h>
#endif

#define FILTER_TABLE_SIZE 16

enum ImageDrawType
{
	Normal,
	HDR,
	BlockVariance,
	PixelVariance,
	BlockPixelVariance,
	AverageVariance,
	BlockAverageVariance,
};

class ID3D11Resource;

class e_Image
{
public:
    // ImageFilm Public Methods
	CUDA_FUNC_IN e_Image(){}

	e_Image(int xRes, int yRes, unsigned int viewGLTexture);
#ifdef ISWINDOWS
	e_Image(int xRes, int yRes, ID3D11Resource *pD3DResource);
#endif
	e_Image(int xRes, int yRes, RGBCOL* target = 0);
    void Free();
	void getExtent(unsigned int& xRes, unsigned int &yRes)
	{
		xRes = xResolution;
		yRes = yResolution;
	}
	CUDA_DEVICE CUDA_HOST void AddSample(int sx, int sy, const Spectrum &L);
	void setStdFilter()
	{
		e_KernelFilter flt;
		flt.SetData(e_KernelBoxFilter(1,1));
		setFilter(flt);
	}
	void setFilter(const e_KernelFilter& filt)
	{
		filter = filt;
		rebuildFilterTable();
	}
	CUDA_DEVICE CUDA_HOST void SetSample(int sx, int sy, RGBCOL c);
    CUDA_DEVICE CUDA_HOST void Splat(int sx, int sy, const Spectrum &L);
    void WriteDisplayImage(const char* fileName);
	void WriteImage(const char* fileName, float splat);
	void StartRendering();
	void EndRendering();
	void Clear();
	struct Pixel {
        Pixel() {
            xyz[0] = xyz[1] = xyz[2] = 0;
			xyzSplat[0] = xyzSplat[1] = xyzSplat[2] = 0;
            weightSum = 0.0f;
			I = I2 = 0.0f;
			E = E2 = 0.0f;
        }
        float xyz[3];
        float weightSum;
        float xyzSplat[3];
		float I, I2;
		float E, E2;
		CUDA_DEVICE CUDA_HOST Spectrum toSpectrum(float splat);
		CUDA_FUNC_IN float var()
		{
			return I2 - I * I;
		}
    };
	e_KernelFilter& accessFilter()
	{
		return filter;
	}
	void rebuildFilterTable();
	ImageDrawType& accessDrawStyle()
	{
		return drawStyle;
	}
	void DoUpdateDisplay(float splat);
	RGBCOL* getCudaPixels(){return viewTarget;}
	void calculateBlockVariance(int block, float splatScale, float* deviceBuffer);
private:
	unsigned int NumFrame;
	void InternalUpdateDisplay(float splat);
	bool m_bDoUpdate;
	e_KernelFilter filter;
    Pixel *cudaPixels;
	Pixel *hostPixels;
	bool usedHostPixels;
	float filterTable[FILTER_TABLE_SIZE][FILTER_TABLE_SIZE];
	int xResolution, yResolution;
	ImageDrawType drawStyle;
	CUDA_FUNC_IN Pixel* getPixel(int i)
	{
#ifdef ISCUDA
		return cudaPixels + i;
#else
		usedHostPixels = true;
		return hostPixels + i;
#endif
	}

	//opengl
	int outState;

	cudaGraphicsResource_t viewCudaResource;
	bool isMapped;
	cudaArray_t viewCudaArray;
	cudaSurfaceObject_t viewCudaSurfaceObject;

	bool ownsTarget;
	RGBCOL* viewTarget;


};