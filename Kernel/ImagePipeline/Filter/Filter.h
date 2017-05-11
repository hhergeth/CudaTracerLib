#pragma once

#include <Engine/Image.h>
#include <Kernel/PixelVarianceBuffer.h>
#include <Kernel/TracerSettings.h>

namespace CudaTracerLib
{

class ImageSamplesFilter
{
protected:
	TracerParameterCollection m_settings;
public:
	virtual ~ImageSamplesFilter()
	{

	}
	virtual void Free() = 0;
	virtual void Resize(int xRes, int yRes) = 0;
	virtual void Apply(Image& img, int numPasses, float splatScale, const PixelVarianceBuffer& varBuffer) = 0;
	virtual TracerParameterCollection& getParameterCollection()
	{
		return m_settings;
	}
};

}