#pragma once
#include <Engine/Image.h>
#include <Engine/SynchronizedBuffer.h>
#include <Math/VarAccumulator.h>

namespace CudaTracerLib
{

struct PixelVarianceInfo
{
	//variance of the iterations of the estimator
	VarAccumulator<float> I;
	//variance of the estimator progressing over iterations
	VarAccumulator<float> E_I;

	CUDA_FUNC_IN void updateMoments(const PixelData& pixel, float numPasses, float splatScale)
	{
		Spectrum s, s2;
		s.fromLinearRGB(pixel.rgb[0], pixel.rgb[1], pixel.rgb[2]);
		s2.fromLinearRGB(pixel.rgbSplat[0], pixel.rgbSplat[1], pixel.rgbSplat[2]);
		Spectrum pixel_color = s + s2 * splatScale;//value of pixel after iteration
		float SUM_after = pixel_color.getLuminance();

		//update moments of the estimator iterations
		float estimator_value = SUM_after - I.E(numPasses);
		I += estimator_value;

		//update moments of the progressing estimator
		float E_N = pixel.toSpectrum(splatScale).getLuminance();
		E_I += E_N;
	}
};


class PixelVarianceBuffer
{
	SynchronizedBuffer<PixelVarianceInfo> m_pixelBuffer;
	unsigned int width, height;
	unsigned int m_numPasses;
public:
	PixelVarianceBuffer(unsigned int w, unsigned int h)
		: m_pixelBuffer(w * h), width(w), height(h)
	{

	}

	void Free()
	{
		m_pixelBuffer.Free();
	}

	void Clear()
	{
		m_pixelBuffer.Memset(0);
	}

	void AddPass(Image& img, float splatScale);

	CUDA_FUNC_IN PixelVarianceInfo& operator()(unsigned int x, unsigned int y)
	{
		return m_pixelBuffer[y * width + x];
	}

	CUDA_FUNC_IN const PixelVarianceInfo& operator()(unsigned int x, unsigned int y) const
	{
		return ((PixelVarianceBuffer*)this)->operator()(x, y);
	}
};

}