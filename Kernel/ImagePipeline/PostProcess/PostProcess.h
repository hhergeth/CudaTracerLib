#pragma once

#include <Engine/Image.h>

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
	virtual void Apply(Image& img, int numPasses) = 0;
};

}