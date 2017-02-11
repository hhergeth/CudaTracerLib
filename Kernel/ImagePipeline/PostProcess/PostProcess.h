#pragma once

#include <Engine/Image.h>
#include <Kernel/PixelVarianceBuffer.h>

namespace CudaTracerLib
{

class PostProcess
{
public:
	virtual ~PostProcess()
	{

	}
	virtual void Free() = 0;
	virtual void Resize(int xRes, int yRes) = 0;
	virtual void Apply(Image& img, int numPasses, const PixelVarianceBuffer& varBuffer) = 0;
};

}