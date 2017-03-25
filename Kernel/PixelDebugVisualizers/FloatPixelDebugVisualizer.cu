#include "FloatPixelDebugVisualizer.h"
#include <Engine/Image.h>
#include "PixelDebugVisualizerHelpers.h"
#include <Kernel/TraceHelper.h>

namespace CudaTracerLib {

struct float_op
{
	CUDA_DEVICE void operator()(unsigned int x, unsigned int y, Image& img, PixelDebugVisualizer<float>& buffer, normalized_data<float>& arg)
	{
		auto val = buffer.getScaledValue(x, y);
		Spectrum col;
		if (arg.normalize)
		{
			val = (val - arg.min) / (arg.max - arg.min);
			col = colorize(val);
		}
		else col = val;

		img.getProcessedData(x, y) = col.toRGBCOL();
	}
};

void PixelDebugVisualizer<float>::Visualize(Image& img)
{
	m_buffer.Synchronize();

	auto arg = generate_normalize_data(*this, img);

	Launch(img, *this, arg, float_op());
}

void PixelDebugVisualizer<float>::VisualizePixel(unsigned int x, unsigned int y, const IDebugDrawer& drawer)
{
	auto prim_ray = g_SceneData.GenerateSensorRay(x, y);
	auto res = traceRay(prim_ray);
	if (!res.hasHit())
		return;

	DifferentialGeometry dg;
	res.fillDG(dg);

	auto v = getScaledValue(x, y);

	if (m_pixelType == VisualizePixelType::Normal)
	{
		drawer.DrawLine(dg.P, dg.P + dg.sys.toWorld(Vec3f(0.0f, 0.0f, v)));
	}
	else if (m_pixelType == VisualizePixelType::Circle)
	{
		drawer.DrawEllipse(dg.P, dg.sys.t, dg.sys.s, v, v);
	}
	else if (m_pixelType == VisualizePixelType::NormalCone)
	{
		drawer.DrawCone(dg.P, dg.sys.n, math::clamp(v, 0.0f, 2.0f * PI), m_coneScale);
	}
}

}