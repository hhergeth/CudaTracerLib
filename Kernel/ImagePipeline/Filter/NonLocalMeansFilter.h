#pragma once

#include "Filter.h"
#include <Engine/Filter.h>
#include <Math/half.h>
#include <Math/Compression.h>

namespace CudaTracerLib
{

class NonLocalMeansFilter : public ImageSamplesFilter
{
public:
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

		CUDA_FUNC_IN static float w_aux(const FeatureData& lhs, const FeatureData& rhs)
		{
			auto d_f_i = [](auto a, auto b)
			{
				return math::sqr(a - b).sum();
			};

			Spectrum sa, sb;
			sa.fromRGBCOL(lhs.m_albedo); sb.fromRGBCOL(rhs.m_albedo);
			float r = math::exp(-d_f_i(sa, sb));
			//r *= math::exp(-d_f_i(Uchar2ToNormalizedFloat3(lhs.m_normal), Uchar2ToNormalizedFloat3(lhs.m_normal)));
			r = math::exp(-d_f_i(Vec2f(lhs.m_depth.ToFloat(), 0), Vec2f(rhs.m_depth.ToFloat(), 0))*500);
			return r;
		}
	};
private:
	RGBE* m_cachedImg;
	FeatureData* m_featureBuffer;
public:
	NonLocalMeansFilter()
		: m_cachedImg(0)
	{
	}
	virtual void Free()
	{
		if (m_cachedImg)
		{
			CUDA_FREE(m_cachedImg);
			CUDA_FREE(m_featureBuffer);
		}
	}
	virtual void Resize(int xRes, int yRes)
	{
		Free();
		CUDA_MALLOC(&m_cachedImg, xRes * yRes * sizeof(RGBE));
		CUDA_MALLOC(&m_featureBuffer, xRes * yRes * sizeof(FeatureData));
	}
	virtual void Apply(Image& img, int numPasses, float splatScale, const PixelVarianceBuffer& varBuffer);
};

}