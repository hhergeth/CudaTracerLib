#pragma once

#include <Math/Spectrum.h>
#include <Base/SynchronizedBuffer.h>

struct FIBITMAP;

namespace CudaTracerLib {

struct PixelData
{
	CUDA_FUNC_IN PixelData()
	{
		rgb[0] = rgb[1] = rgb[2] = 0;
		rgbSplat[0] = rgbSplat[1] = rgbSplat[2] = 0;
		weightSum = 0.0f;
	}
	float rgb[3];
	float rgbSplat[3];
	float weightSum;
	CUDA_FUNC_IN Spectrum toSpectrum(float splatScale) const
	{
		float weight = weightSum != 0 ? weightSum : 1;
		Spectrum s, s2;
		s.fromLinearRGB(rgb[0], rgb[1], rgb[2]);
		s2.fromLinearRGB(rgbSplat[0], rgbSplat[1], rgbSplat[2]);
		return (s / weight + s2 * splatScale);
	}
};

class Image : public ISynchronizedBufferParent
{
public:
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
	CTL_EXPORT CUDA_DEVICE CUDA_HOST void ClearSample(int sx, int sy);
	CUDA_DEVICE void SetSample(int sx, int sy, RGBCOL c);
	CTL_EXPORT CUDA_DEVICE CUDA_HOST void Splat(float sx, float sy, const Spectrum &L);

	CTL_EXPORT void WriteDisplayImage(const std::string& fileName);
	CTL_EXPORT void SaveToMemory(void** mem, size_t& size, const std::string& type);

	CTL_EXPORT void Clear();

	//compute the maximum, minimum, average luminance and log average luminance of the filtered data
	CTL_EXPORT void ComputeLuminanceInfo(Spectrum& avgColor, float& minLum, float& maxLum, float& avgLum, float& avgLogLum);

	CUDA_FUNC_IN PixelData& getPixelData(int x, int y)
	{
		return m_pixelBuffer[idx(x, y)];
	}
	CUDA_FUNC_IN RGBE& getFilteredData(int x, int y)
	{
		return m_filteredColorsDevice[idx(x, y)];
	}
	CUDA_FUNC_IN RGBCOL& getProcessedData(int x, int y)
	{
		return m_viewTarget[idx(x, y)];
	}
private:
	FIBITMAP* toFreeImage(bool HDR);
	CUDA_FUNC_IN int idx(int x, int y) const
	{
		return y * xResolution + x;
	}

	int xResolution, yResolution;
	//Stage 1, directly from the Integrator, this is either on the host or device
	SynchronizedBuffer<PixelData> m_pixelBuffer;
	//Stage 2, reconstructed from Stage 1 by a Filter, located on device
	RGBE* m_filteredColorsDevice;
	//Stage 3, applied some sort of color transform to Stage 2, located on device
	bool ownsTarget;
	RGBCOL* m_viewTarget;
};

}
