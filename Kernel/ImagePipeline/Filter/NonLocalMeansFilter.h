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