#pragma once
#include "PixelDebugVisualizer.h"

namespace CudaTracerLib {

template<> class PixelDebugVisualizer<Vec2f> : public PixelDebugVisualizerBase<Vec2f>
{
public:
	enum class VisualizePixelType
	{
		Ellipse,
		//visualizes the value as element of the tangent plane
		OnSurface,
		//assumes v = (theta, r) and visualizes that in the tangent plane
		PolarCoordinates,
		//assumes v = (theta, phi) and visualizes that in the 2-sphere with unit radius
		SphericalCoordinates,
	};
	VisualizePixelType m_pixelType;
public:
	PixelDebugVisualizer(const std::string& name)
		: PixelDebugVisualizerBase(name), m_pixelType(VisualizePixelType::Ellipse)
	{

	}

	virtual void Visualize(Image& img);
	virtual void VisualizePixel(unsigned int x, unsigned int y, const IDebugDrawer& drawer);
};

}