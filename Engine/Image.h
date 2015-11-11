#pragma once

#include "Filter.h"

#ifdef ISWINDOWS
struct ID3D11Resource;
#endif

struct FIBITMAP;

namespace CudaTracerLib {

enum ImageDrawType
{
	Normal,
	HDR,
};

class Image
{
public:
	CUDA_FUNC_IN Image(){}

	Image(int xRes, int yRes, unsigned int viewGLTexture);
#ifdef ISWINDOWS
	Image(int xRes, int yRes, ID3D11Resource *pD3DResource);
#endif
	Image(int xRes, int yRes, RGBCOL* target = 0);
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
		Filter flt;
		flt.SetData(BoxFilter(1, 1));
		setFilter(flt);
	}
	void setFilter(const Filter& filt)
	{
		filter = filt;
	}
	CUDA_DEVICE CUDA_HOST void SetSample(int sx, int sy, RGBCOL c);
	CUDA_DEVICE CUDA_HOST void Splat(float sx, float sy, const Spectrum &L);
	void WriteDisplayImage(const std::string& fileName);
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
	Filter& accessFilter()
	{
		return filter;
	}
	ImageDrawType& accessDrawStyle()
	{
		return drawStyle;
	}
	void DoUpdateDisplay(float splat);
	RGBCOL* getCudaPixels(){ return viewTarget; }
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
	void copyToHost();
	void SaveToMemory(void** mem, size_t& size, const std::string& type);
	static void ComputeDiff(const Image& A, const Image& B, Image& dest, float scale);
	void setOutputScale(float f){ m_fOutScale = f; }
private:
	FIBITMAP* toFreeImage();
	void InternalUpdateDisplay();

	float m_fOutScale;
	bool m_bDoUpdate;
	Filter filter;
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

}