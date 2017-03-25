#include "FloatPixelDebugVisualizer.h"
#include <Engine/Image.h>
#include "PixelDebugVisualizerHelpers.h"
#include <Kernel/TraceHelper.h>

namespace CudaTracerLib {

struct normalized_data
{
	bool normalize;
	float min, max;
};
struct float_op
{
	CUDA_DEVICE void operator()(unsigned int x, unsigned int y, Image& img, PixelDebugVisualizer<float>& buffer, normalized_data& arg)
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
CUDA_DEVICE int g_min, g_max;
struct normalize_float_op
{
	CUDA_DEVICE void operator()(unsigned int x, unsigned int y, Image& img, PixelDebugVisualizer<float>& buffer, int)
	{
		float val = buffer.getScaledValue(x, y);
		CUDA_SHARED int s_min, s_max;
		if (threadIdx.x == 0 && threadIdx.y == 0)
		{
			s_min = INT_MAX;
			s_max = -INT_MAX;
		}
		__syncthreads();

		auto int_val = floatToOrderedInt(val);
		atomicMin(&g_min, int_val);
		atomicMax(&g_max, int_val);

		if (threadIdx.x == 0 && threadIdx.y == 0)
		{
			atomicMin(&g_min, s_min);
			atomicMax(&g_max, s_max);
		}
	}
};
void PixelDebugVisualizer<float>::Visualize(Image& img)
{
	m_buffer.Synchronize();

	normalized_data arg;
	arg.normalize = m_normalize;
	if (arg.normalize)
	{
		int min_ = floatToOrderedInt(FLT_MAX), max_ = floatToOrderedInt(-FLT_MAX);
		CopyToSymbol(g_min, min_);
		CopyToSymbol(g_max, max_);
		Launch(img, *this, -1, normalize_float_op());
		CopyFromSymbol(g_min, min_);
		CopyFromSymbol(g_max, max_);
		arg.min = orderedIntToFloat(min_);
		arg.max = orderedIntToFloat(max_);
	}

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
		drawer.DrawLine(dg.P, dg.P + dg.sys.n * v);
	}
	else if (m_pixelType == VisualizePixelType::Circle)
	{
		drawer.DrawEllipseOnSurface(dg.P, dg.sys.t, dg.sys.s, v, v);
	}
}

}