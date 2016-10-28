#pragma once

#include "PostProcess.h"

namespace CudaTracerLib {

class ToneMapPostProcess : public PostProcess
{
public:
	ToneMapPostProcess()
	{

	}
	virtual void Free()
	{

	}
	virtual void Resize(int xRes, int yRes)
	{

	}
	virtual void Apply(Image& img, int numPasses);
};

}