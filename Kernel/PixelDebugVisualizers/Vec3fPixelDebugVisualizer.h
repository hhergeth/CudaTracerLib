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
		Elipsoid,
		//visualizes the value as element of the orthonormal surface base
		OnSurface,
	};
	VisualizePixelType m_pixelType;
public:
	PixelDebugVisualizer(const std::string& name)
		: PixelDebugVisualizerBase(name), m_normalize(false), m_pixelType(VisualizePixelType::Elipsoid)
	{

	}

	virtual void Visualize(Image& img);
	virtual void VisualizePixel(unsigned int x, unsigned int y, const IDebugDrawer& drawer);
};

}