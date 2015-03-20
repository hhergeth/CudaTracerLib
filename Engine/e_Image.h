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
	CUDA_FUNC_IN void getExtent(unsigned int& xRes, unsigned int &yRes) const
	{
		xRes = xResolution;
		yRes = yResolution;
	}
	CUDA_FUNC_IN unsigned int getWidth()
	{
		return xResolution;
	}
	CUDA_FUNC_IN unsigned int getHeight()
	{
		return yResolution;
	}
	CUDA_DEVICE CUDA_HOST void AddSample(float sx, float sy, const Spectrum &L);
	CUDA_FUNC_IN void ClearSample(int sx, int sy)
	{
		*getPixel(sy * xResolution + sx) = Pixel();
	}
	void setStdFilter()
	{
		e_Filter flt;
		flt.SetData(e_BoxFilter(1,1));
		setFilter(flt);
	}
	void setFilter(const e_Filter& filt)
	{
		filter = filt;
	}
	CUDA_DEVICE CUDA_HOST void SetSample(int sx, int sy, RGBCOL c);
	CUDA_DEVICE CUDA_HOST void Splat(float sx, float sy, const Spectrum &L);
    void WriteDisplayImage(const char* fileName);
	void StartRendering();
	void EndRendering();
	void Clear();
	struct Pixel {
        CUDA_FUNC_IN Pixel() {
			rgb[0] = rgb[1] = rgb[2] = 0;
			rgbSplat[0] = rgbSplat[1] = rgbSplat[2] = 0;
            weightSum = 0.0f;
        }
        float rgb[3];
        float weightSum;
		float rgbSplat[3];
		CUDA_FUNC_IN Spectrum toSpectrum(float splatScale)
		{
			float weight = weightSum != 0 ? weightSum : 1;
			Spectrum s, s2;
			s.fromLinearRGB(rgb[0], rgb[1], rgb[2]);
			s2.fromLinearRGB(rgbSplat[0], rgbSplat[1], rgbSplat[2]);
			return (s / weight + s2 * splatScale);
		}
    };
	e_Filter& accessFilter()
	{
		return filter;
	}
	ImageDrawType& accessDrawStyle()
	{
		return drawStyle;
	}
	void DoUpdateDisplay(float splat);
	RGBCOL* getCudaPixels(){return viewTarget;}
	CUDA_FUNC_IN Spectrum getPixel(int x, int y)
	{
		return getPixel(y * xResolution + x)->toSpectrum(lastSplatVal);
	}
	CUDA_FUNC_IN Pixel& accessPixel(int x, int y)
	{
		return *getPixel(y * xResolution + x);
	}
	void DrawSamplePlacement(int numPasses);
	void disableUpdate()
	{
		m_bDoUpdate = false;
	}
	void copyToHost()
	{
		cudaMemcpy(hostPixels, cudaPixels, sizeof(Pixel) * xResolution * yResolution, cudaMemcpyDeviceToHost);
	}
	static void ComputeDiff(const e_Image& A, const e_Image& B, e_Image& dest, float scale);
private:
	void InternalUpdateDisplay();
	bool m_bDoUpdate;
	e_Filter filter;
    Pixel *cudaPixels;
	Pixel *hostPixels;
	bool usedHostPixels;
	int xResolution, yResolution;
	ImageDrawType drawStyle;
	float lastSplatVal;
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