#include "Vec2fPixelDebugVisualizer.h"
#include <Engine/Image.h>
#include "PixelDebugVisualizerHelpers.h"
#include <Kernel/TraceHelper.h>
#include <Math/Sampling.h>

namespace CudaTracerLib {

struct Vec2f_op
{
	CUDA_DEVICE void operator()(unsigned int x, unsigned int y, Image& img, PixelDebugVisualizer<Vec2f>& buffer, bool arg)
	{
		auto val = buffer.getScaledValue(x, y);
		Spectrum col;
		if (arg)
			col = Spectrum((val.x + 1) / 2, (val.y + 1) / 2, 0.0f);
		else col = Spectrum(val.x, val.y, 0.0f);

		img.getProcessedData(x, y) = col.toRGBCOL();
	}
};

void PixelDebugVisualizer<Vec2f>::Visualize(Image& img)
{
	m_buffer.Synchronize();

	Launch(img, *this, m_normalize, Vec2f_op());
}


void PixelDebugVisualizer<Vec2f>::VisualizePixel(unsigned int x, unsigned int y, const IDebugDrawer& drawer)
{
	auto prim_ray = g_SceneData.GenerateSensorRay(x, y);
	auto res = traceRay(prim_ray);
	if (!res.hasHit())
		return;

	DifferentialGeometry dg;
	res.fillDG(dg);

	auto v = getScaledValue(x, y);

	if (m_pixelType == VisualizePixelType::OnSurface)
	{
		drawer.DrawLine(dg.P, dg.P + dg.sys.toWorld(Vec3f(v.x, v.y, 0.0f)));
	}
	else if (m_pixelType == VisualizePixelType::Ellipse)
	{
		drawer.DrawEllipse(dg.P, dg.sys.t, dg.sys.s, v.x, v.y);
	}
	else if (m_pixelType == VisualizePixelType::PolarCoordinates)
	{
		Vec2f t(v.y * math::cos(v.x), v.y * math::sin(v.x));
		drawer.DrawLine(dg.P, dg.P + dg.sys.toLocal(Vec3f(t.x, t.y, 0.0f)));
	}
	else if (m_pixelType == VisualizePixelType::SphericalCoordinates)
	{
		auto dir = MonteCarlo::SphericalDirection(v.x, v.y);
		drawer.DrawLine(dg.P, dg.P + dg.sys.toWorld(dir));
	}
}

}