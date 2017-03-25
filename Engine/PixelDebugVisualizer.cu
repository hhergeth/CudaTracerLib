#include "PixelDebugVisualizer.h"
#include "Image.h"

namespace CudaTracerLib {

CUDA_FUNC_IN Spectrum colorize(float t)
{
	Spectrum qs;
	qs.fromHSL(2.0f / 3.0f * (1 - math::clamp01(t)), 1, 0.5f);//0 -> 1 : Dark Blue -> Light Blue -> Green -> Yellow -> Red
	return qs;
}

template<typename ARG, typename F, typename T> CUDA_GLOBAL void generic_kernel(Image& img, PixelDebugVisualizer<T> buffer, ARG arg, F clb)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x < img.getWidth() && y < img.getHeight())
		clb(x, y, img, buffer, arg);
}

template<typename ARG, typename F, typename T> void Launch(Image& img, PixelDebugVisualizer<T>& buffer, const ARG& arg, F clb)
{
	const int block = 32;
	generic_kernel<<< dim3(img.getWidth() / block + 1, img.getHeight() / block + 1), dim3(block, block) >>>(img, buffer, arg, clb);
}

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


void PixelDebugVisualizer<float>::VisualizePixel(const IDebugDrawer& drawer)
{

}

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


void PixelDebugVisualizer<Vec2f>::VisualizePixel(const IDebugDrawer& drawer)
{

}

struct Vec3f_op
{
	CUDA_DEVICE void operator()(unsigned int x, unsigned int y, Image& img, PixelDebugVisualizer<Vec3f>& buffer, bool arg)
	{
		auto val = buffer.getScaledValue(x, y);
		Spectrum col;
		if (arg)
			col = Spectrum((val.x + 1) / 2, (val.y + 1) / 2, (val.z + 1) / 2);
		else col = Spectrum(val.x, val.y, val.z);

		img.getProcessedData(x, y) = col.toRGBCOL();
	}
};

void PixelDebugVisualizer<Vec3f>::Visualize(Image& img)
{
	m_buffer.Synchronize();

	Launch(img, *this, m_normalize, Vec3f_op());
}


void PixelDebugVisualizer<Vec3f>::VisualizePixel(const IDebugDrawer& drawer)
{

}

void IPixelDebugVisualizer::VisualizeFeatures(const IDebugDrawer& drawer, IPixelDebugVisualizer::FeatureVisualizer features)
{

}

}