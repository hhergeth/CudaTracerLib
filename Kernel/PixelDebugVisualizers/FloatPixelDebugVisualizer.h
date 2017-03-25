#pragma once
#include "PixelDebugVisualizer.h"

namespace CudaTracerLib {

template<> class PixelDebugVisualizer<float> : public PixelDebugVisualizerBase<float>
{
public:
	//linear normalization from [a,b] -> [0, 1]
	bool m_normalize;
	enum class VisualizePixelType
	{
		Circle,
		//visualizes the value as scaled normal
		Normal,
	};
	VisualizePixelType m_pixelType;
public:
	PixelDebugVisualizer(const std::string& name)
		: PixelDebugVisualizerBase(name), m_normalize(true), m_pixelType(VisualizePixelType::Circle)
	{

	}

	virtual void Visualize(Image& img);
	virtual void VisualizePixel(unsigned int x, unsigned int y, const IDebugDrawer& drawer);
};

}