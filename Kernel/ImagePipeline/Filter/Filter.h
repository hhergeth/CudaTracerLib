#pragma once

#include <Engine/Image.h>

namespace CudaTracerLib
{

class ImageSamplesFilter
{
public:
	virtual ~ImageSamplesFilter()
	{

	}
	virtual void Free() = 0;
	virtual void Resize(int xRes, int yRes) = 0;
	virtual void Apply(Image& img, int numPasses, float splatScale) = 0;
};

}