#pragma once

#include "Filter.h"
#include <SceneTypes/Filter.h>
#include <Math/half.h>
#include <Math/Compression.h>

namespace CudaTracerLib
{

class NonLocalMeansFilter : public ImageSamplesFilter
{
public:
	struct PixelFilterWeightBuffer
	{
		float* weights;
		int R;

		CUDA_FUNC_IN PixelFilterWeightBuffer(float* weights, int R)
			: weights(weights), R(R)
		{

		}

		CUDA_FUNC_IN float& operator()(int lx, int ly)
		{
			lx += R;
			ly += R;
			return weights[ly * (2 * R + 1) + lx];
		}
	};
	struct FilterWeightBuffer
	{
		int R, F, w, h;
		int n_weights_per_pixel;
		float* deviceWeightBuffer;

		FilterWeightBuffer()
			: R(-1), F(-1), w(-1), h(-1), n_weights_per_pixel(-1), deviceWeightBuffer(0)
		{

		}

		void Free()
		{
			if (deviceWeightBuffer)
				CUDA_FREE(deviceWeightBuffer);
		}

		void ClearBuffer()
		{
			cudaMemset(deviceWeightBuffer, 0, w * h * n_weights_per_pixel * sizeof(float));
		}

		bool canUseBuffer(int r, int f)
		{
			return r == R && f == F;
		}

		void adaptBuffer(int r, int f, int _w, int _h)
		{
			R = r;
			F = f;
			w = _w;
			h = _h;
			n_weights_per_pixel = math::sqr(2 * R + 1);
			Free();
			CUDA_MALLOC(&deviceWeightBuffer, w * h * n_weights_per_pixel * sizeof(float));
		}

		CUDA_FUNC_IN PixelFilterWeightBuffer operator()(int x, int y) const
		{
			return PixelFilterWeightBuffer(deviceWeightBuffer + (y * w + x) * n_weights_per_pixel, R);
		}
	};

	struct FeatureData
	{
		half m_depth;
		unsigned short m_normal;
		RGBCOL m_albedo;

		CUDA_FUNC_IN FeatureData()
		{

		}

		CUDA_FUNC_IN static FeatureData Init()
		{
			FeatureData f;
			f.m_albedo = make_uchar4(0, 0, 0, 0);
			f.m_depth = 0.0f;
			f.m_normal = 0;
			return f;
		}
	};
private:
	RGBE* m_cachedImg;
	FeatureData* m_featureBuffer;
	FilterWeightBuffer m_weightBuffer;
	int last_iter_weight_update;
public:
	PARAMETER_KEY(float, k)
	PARAMETER_KEY(float, sigma2Scale)
	PARAMETER_KEY(int, UpdateWeightPeriodicity)

	NonLocalMeansFilter()
		: m_cachedImg(0), last_iter_weight_update(-1)
	{
		m_settings	<< KEY_k()								<< CreateInterval(0.45f, 0.0f, FLT_MAX)
					<< KEY_sigma2Scale()					<< CreateInterval(0.005f, 0.0f, FLT_MAX)
					<< KEY_UpdateWeightPeriodicity()		<< CreateInterval(25, 0, INT_MAX);
	}
	virtual void Free()
	{
		if (m_cachedImg)
		{
			CUDA_FREE(m_cachedImg);
			CUDA_FREE(m_featureBuffer);
		}
		m_weightBuffer.Free();
	}
	virtual void Resize(int xRes, int yRes)
	{
		Free();
		CUDA_MALLOC(&m_cachedImg, xRes * yRes * sizeof(RGBE));
		CUDA_MALLOC(&m_featureBuffer, xRes * yRes * sizeof(FeatureData));
		last_iter_weight_update = -1;
	}
	virtual void Apply(Image& img, int numPasses, float splatScale, const PixelVarianceBuffer& varBuffer);
};

}