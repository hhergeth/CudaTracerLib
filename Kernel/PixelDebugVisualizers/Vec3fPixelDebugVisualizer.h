#pragma once
#include "PixelDebugVisualizer.h"

namespace CudaTracerLib {

template<> class PixelDebugVisualizer<Vec3f> : public PixelDebugVisualizerBase<Vec3f>
{
public:
	//linear normalization from [-1,1] -> [0, 1]
	bool m_normalize;
	enum class VisualizePixelType
	{
		//visualizes the value as world space vector
		Vector,
		//uses the value as axis lengths for an ellipsoid
		Ellipsoid,
		//visualizes the value as element of the orthonormal surface base
		OnSurface,
		//assumes value = (theta, phi, r) and visualizes that as element of the 2-ball
		SphericalCoordinates_World,
		SphericalCoordinates_Local,
	};
	VisualizePixelType m_pixelType;
public:
	PixelDebugVisualizer(const std::string& name)
		: PixelDebugVisualizerBase(name), m_normalize(false), m_pixelType(VisualizePixelType::Ellipsoid)
	{

	}

	virtual void Visualize(Image& img);
	virtual void VisualizePixel(unsigned int x, unsigned int y, const IDebugDrawer& drawer);
};

}