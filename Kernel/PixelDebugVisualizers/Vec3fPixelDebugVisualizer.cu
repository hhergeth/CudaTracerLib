#include "Vec3fPixelDebugVisualizer.h"
#include <Engine/Image.h>
#include "PixelDebugVisualizerHelpers.h"
#include <Kernel/TraceHelper.h>
#include <Math/Warp.h>

namespace CudaTracerLib {

struct Vec3f_op
{
	CUDA_DEVICE void operator()(unsigned int x, unsigned int y, Image& img, PixelDebugVisualizer<Vec3f>& buffer, normalized_data<Vec3f>& arg)
	{
		auto val = buffer.getScaledValue(x, y);
		Spectrum col;
		if (arg.normalize)
		{
			auto nor = (val - arg.min) / (arg.max - arg.min);
			col = Spectrum(nor.x, nor.y, nor.z);
		}
		else col = Spectrum(val.x, val.y, val.z);

		img.getProcessedData(x, y) = col.toRGBCOL();
	}
};

void PixelDebugVisualizer<Vec3f>::Visualize(Image& img)
{
	m_buffer.Synchronize();

	auto arg = generate_normalize_data(*this, img);

	Launch(img, *this, arg, Vec3f_op());
}

void PixelDebugVisualizer<Vec3f>::VisualizePixel(unsigned int x, unsigned int y, const IDebugDrawer& drawer)
{
	auto prim_ray = g_SceneData.GenerateSensorRay(x, y);
	auto res = traceRay(prim_ray);
	if (!res.hasHit())
		return;

	DifferentialGeometry dg;
	dg.P = prim_ray(res.m_fDist);
	res.fillDG(dg);

	auto v = getScaledValue(x, y);

	if (m_pixelType == VisualizePixelType::Vector)
	{
		drawer.DrawLine(dg.P, dg.P + v);
	}
	else if (m_pixelType == VisualizePixelType::OnSurface)
	{
		drawer.DrawLine(dg.P, dg.P + dg.sys.toWorld(v));
	}
	else if (m_pixelType == VisualizePixelType::Ellipsoid)
	{
		drawer.DrawEllipsoid(dg.P, dg.sys.s, dg.sys.t, dg.sys.n, v.x, v.y, v.z);
	}
	else if (m_pixelType == VisualizePixelType::SphericalCoordinates_World || m_pixelType == VisualizePixelType::SphericalCoordinates_Local)
	{
		auto dir = Warp::SphericalDirection(v.x, v.y);
		drawer.DrawLine(dg.P, dg.P + (m_pixelType == VisualizePixelType::SphericalCoordinates_World ? dir : dg.sys.toWorld(dir)) * v.z);
	}
}

}