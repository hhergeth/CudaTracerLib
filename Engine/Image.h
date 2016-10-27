#pragma once

#include "Filter.h"
#include <Math/Spectrum.h>

struct FIBITMAP;

namespace CudaTracerLib {

class Image
{
public:
	CUDA_FUNC_IN Image(){}

	CTL_EXPORT Image(int xRes, int yRes, RGBCOL* target = 0);
	CTL_EXPORT void Free();
	CUDA_FUNC_IN void getExtent(unsigned int& xRes, unsigned int &yRes) const
	{
		xRes = xResolution;
		yRes = yResolution;
	}
	CUDA_FUNC_IN unsigned int getWidth() const
	{
		return xResolution;
	}
	CUDA_FUNC_IN unsigned int getHeight() const
	{
		return yResolution;
	}
	CTL_EXPORT CUDA_DEVICE CUDA_HOST void AddSample(float sx, float sy, const Spectrum &L);
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
	CTL_EXPORT CUDA_DEVICE CUDA_HOST void SetSample(int sx, int sy, RGBCOL c);
	CTL_EXPORT CUDA_DEVICE CUDA_HOST void Splat(float sx, float sy, const Spectrum &L);
	CTL_EXPORT void WriteDisplayImage(const std::string& fileName);
	CTL_EXPORT void StartRendering();
	CTL_EXPORT void EndRendering();
	CTL_EXPORT void Clear();
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
	CTL_EXPORT void DoUpdateDisplay(float splat);
	RGBCOL* getCudaPixels(){ return viewTarget; }
	CUDA_FUNC_IN Spectrum getPixel(int x, int y)
	{
		return getPixel(y * xResolution + x)->toSpectrum(lastSplatVal);
	}
	CUDA_FUNC_IN Pixel& accessPixel(int x, int y)
	{
		return *getPixel(y * xResolution + x);
	}
	void disableUpdate()
	{
		m_bDoUpdate = false;
	}
	CTL_EXPORT void copyToHost();
	CTL_EXPORT void SaveToMemory(void** mem, size_t& size, const std::string& type);
	CTL_EXPORT void setOutputScale(float f){ m_fOutScale = f; }
	RGBE* getFilteredColorsDevice() const
	{
		return m_filteredColorsDevice;
	}
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
	float lastSplatVal;
	RGBE* m_filteredColorsDevice;
	CUDA_FUNC_IN Pixel* getPixel(int i)
	{
#ifdef ISCUDA
		return cudaPixels + i;
#else
		usedHostPixels = true;
		return hostPixels + i;
#endif
	}

	bool ownsTarget;
	RGBCOL* viewTarget;
};

}
