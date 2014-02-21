#pragma once

#include "e_Filter.h"
#include "e_Samples.h"
#ifdef ISWINDOWS
#include <windows.h>
#endif
#include <cuda_gl_interop.h>

#define FILTER_TABLE_SIZE 16

class e_Image
{
public:
    // ImageFilm Public Methods
	CUDA_FUNC_IN e_Image(){}

	e_Image(int xRes, int yRes, unsigned int viewGLTexture);
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
        }
        float xyz[3];
        float weightSum;
        float xyzSplat[3];
		float I, I2;
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
	bool& accessHDR()
	{
		return doHDR;
	}
	void DoUpdateDisplay(float splat);
	RGBCOL* getCudaPixels(){return viewTarget;}
private:
	void InternalUpdateDisplay(float splat);
	bool m_bDoUpdate;
	e_KernelFilter filter;
    Pixel *cudaPixels;
	Pixel *hostPixels;
	bool usedHostPixels;
	float filterTable[FILTER_TABLE_SIZE][FILTER_TABLE_SIZE];
	int xResolution, yResolution;
	bool doHDR;
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
	unsigned int viewGLTexture;
	cudaGraphicsResource_t viewCudaResource;

	bool isMapped;
	cudaArray_t viewCudaArray;
	cudaSurfaceObject_t viewCudaSurfaceObject;

	bool ownsTarget;
	RGBCOL* viewTarget;
};