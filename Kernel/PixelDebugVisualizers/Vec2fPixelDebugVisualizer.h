#pragma once
#include "PixelDebugVisualizer.h"

namespace CudaTracerLib {

template<> class PixelDebugVisualizer<Vec2f> : public PixelDebugVisualizerBase<Vec2f>
{
public:
	//linear normalization from [-1,1] -> [0, 1]
	bool m_normalize;
	enum class VisualizePixelType
	{
		Ellipse,
		//visualizes the value as element of the tangent plane
		OnSurface,
	};
	VisualizePixelType m_pixelType;
public:
	PixelDebugVisualizer(const std::string& name)
		: PixelDebugVisualizerBase(name), m_normalize(false), m_pixelType(VisualizePixelType::Ellipse)
	{

	}

	virtual void Visualize(Image& img);
	virtual void VisualizePixel(unsigned int x, unsigned int y, const IDebugDrawer& drawer);
};

}