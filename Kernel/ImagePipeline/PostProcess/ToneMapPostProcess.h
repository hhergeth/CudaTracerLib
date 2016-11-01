#pragma once

#include "PostProcess.h"

namespace CudaTracerLib {

//Implementation of Reinhard et. al's Tone Mapping operator
class ToneMapPostProcess : public PostProcess
{
public:
	//settings of the algorithm
	//key controls the target luminance of the image, € [0,1]
	//burn enables regions of the image to "burn" out i.e. be completly white, € [0, 1]
	//for further information please compare with the paper
	float m_key, m_burn;
public:
	ToneMapPostProcess()
		: m_key(0.28f), m_burn(0)
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