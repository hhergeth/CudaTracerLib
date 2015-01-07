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
	HDR
};

struct ID3D11Resource;

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
	void getExtent(unsigned int& xRes, unsigned int &yRes) const
	{
		xRes = xResolution;
		yRes = yResolution;
	}
	CUDA_DEVICE CUDA_HOST void AddSample(float sx, float sy, const Spectrum &L);
	CUDA_FUNC_IN void ClearSample(int sx, int sy)
	{
		*getPixel(sy * xResolution + sx) = Pixel();
	}
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
	CUDA_DEVICE CUDA_HOST void Splat(float sx, float sy, const Spectrum &L);
    void WriteDisplayImage(const char* fileName);
	void WriteImage(const char* fileName);
	void StartRendering();
	void EndRendering();
	void Clear();
	struct Pixel {
        CUDA_FUNC_IN Pixel() {
            xyz[0] = xyz[1] = xyz[2] = 0;
			xyzSplat[0] = xyzSplat[1] = xyzSplat[2] = 0;
            weightSum = 0.0f;
        }
        float xyz[3];
        float weightSum;
        float xyzSplat[3];
		CUDA_DEVICE CUDA_HOST Spectrum toSpectrum(float splat);
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
	CUDA_FUNC_IN Spectrum getPixel(float splat, int x, int y)
	{
		return getPixel(y * xResolution + x)->toSpectrum(splat);
	}
private:
	unsigned int NumFrame;
	void InternalUpdateDisplay(float splat);
	bool m_bDoUpdate;
	e_KernelFilter filter;
    Pixel *cudaPixels;
	Pixel *hostPixels;
	bool usedHostPixels;
	float filterTable[FILTER_TABLE_SIZE * FILTER_TABLE_SIZE];
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