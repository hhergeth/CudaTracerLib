#pragma once
#include "PixelDebugVisualizer.h"

namespace CudaTracerLib {

template<> class PixelDebugVisualizer<float> : public PixelDebugVisualizerBase<float>
{
public:
	enum class VisualizePixelType
	{
		Circle,
		//visualizes the value as scaled normal
		Normal,
		//uses the value as angle of a cone and m_coneScale as the length
		NormalCone,
	};
	VisualizePixelType m_pixelType;
	float m_coneScale;
public:
	PixelDebugVisualizer(const std::string& name)
		: PixelDebugVisualizerBase(name), m_pixelType(VisualizePixelType::Circle), m_coneScale(1)
	{

	}

	virtual void Visualize(Image& img);
	virtual void VisualizePixel(unsigned int x, unsigned int y, const IDebugDrawer& drawer);
};

}