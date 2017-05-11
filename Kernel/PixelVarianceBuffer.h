#pragma once
#include <Engine/Image.h>
#include <Base/SynchronizedBuffer.h>
#include <Math/VarAccumulator.h>

namespace CudaTracerLib
{

struct PixelVarianceInfo
{
	Spectrum prev_I;
	//half buffer which half of the time gets 1 new sample
	Spectrum half_buffer;
	int iterations_done;
	float weight;

	//variance of the iterations of the estimator
	VarAccumulator<float> I;
	int num_samples_var;

	CUDA_FUNC_IN void updateMoments(const PixelData& pixel, float splatScale, float samplerPerformed)
	{
		Spectrum s, s2;
		s.fromLinearRGB(pixel.rgb[0], pixel.rgb[1], pixel.rgb[2]);
		s2.fromLinearRGB(pixel.rgbSplat[0], pixel.rgbSplat[1], pixel.rgbSplat[2]);
		Spectrum new_pixel_sum = s + s2 * splatScale;//value of pixel after iteration
		Spectrum estimator_val_1 = (new_pixel_sum - prev_I) / samplerPerformed;
		prev_I = new_pixel_sum;
		weight = pixel.weightSum;

		if (iterations_done++ % 2 == 1)
		{
			half_buffer += estimator_val_1;
		}

		if (samplerPerformed != 0)
		{
			I += estimator_val_1.getLuminance();
			num_samples_var++;
		}
	}

	CUDA_FUNC_IN float computeVariance() const
	{
		return I.Var((float)num_samples_var);
	}

	CUDA_FUNC_IN float computeAverage() const
	{
		return I.E((float)num_samples_var);
	}

	//error metric from "A Hierarchical Automatic Stopping Condition for Monte Carlo Global Illumination" (2009)
	CUDA_FUNC_IN float computeError() const
	{
		Spectrum I = prev_I / weight;
		Spectrum A = half_buffer / float(iterations_done / 2);
		float e_p = (I - A).abs().sum() / math::sqrt(I.sum());
		return I.isZero() || I.isNaN() || A.isNaN() ? 0.0f : max(e_p, 1e-2f);
	}
};

class IBlockSampler;
class PixelVarianceBuffer : public ISynchronizedBufferParent
{
	SynchronizedBuffer<PixelVarianceInfo> m_pixelBuffer;
	unsigned int width, height;
	unsigned int m_numPasses;
public:
	PixelVarianceBuffer(unsigned int w, unsigned int h)
		: ISynchronizedBufferParent(m_pixelBuffer), m_pixelBuffer(w * h), width(w), height(h)
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

	void AddPass(Image& img, float splatScale, const IBlockSampler* blockSampler);

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