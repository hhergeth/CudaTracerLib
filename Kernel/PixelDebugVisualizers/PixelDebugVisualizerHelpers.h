#pragma once
#include <Math/Vector.h>
#include <Math/Spectrum.h>
#include <Math/MathFunc.h>
#include "PixelDebugVisualizer.h"
//necessary because of cuda compile error
#include "FloatPixelDebugVisualizer.h"

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
	if (x < img.getWidth() && y < img.getHeight())
		clb(x, y, img, buffer, arg);
}

template<typename ARG, typename F, typename T> void Launch(Image& img, PixelDebugVisualizer<T>& buffer, const ARG& arg, F clb)
{
	const int block = 32;
	generic_kernel << < dim3(img.getWidth() / block + 1, img.getHeight() / block + 1), dim3(block, block) >> >(img, buffer, arg, clb);
}

template<int N, typename T> struct generic_normalize_float_op
{
private:
	template<typename T> CUDA_DEVICE float extractComponent(int I, const T& vec)
	{
		return vec[I];
	}
	template<> CUDA_DEVICE float extractComponent(int I, const float& f)
	{
		return f;
	}
public:
	CUDA_DEVICE void operator()(unsigned int x, unsigned int y, Image& img, PixelDebugVisualizer<T>& buffer, int* min_max_buf)
	{
		auto val = buffer.getScaledValue(x, y);
		CUDA_SHARED int s_min[N], s_max[N];
		if (threadIdx.x == 0 && threadIdx.y == 0)
		{
			for (int i = 0; i < N; i++)
			{
				s_min[i] = INT_MAX;
				s_max[i] = -INT_MAX;
			}
		}
		__syncthreads();

		for (int i = 0; i < N; i++)
		{
			auto component_int_val = floatToOrderedInt(extractComponent(i, val));
			atomicMin(&s_min[i], component_int_val);
			atomicMax(&s_max[i], component_int_val);
		}

		if (threadIdx.x == 0 && threadIdx.y == 0)
		{
			for (int i = 0; i < N; i++)
			{
				atomicMin(min_max_buf + i, s_min[i]);
				atomicMax(min_max_buf + N + i, s_max[i]);
			}
		}
	}
};

template<typename T> void compute_range(Image& img, PixelDebugVisualizer<T>& buffer, T& min_res, T& max_res)
{
	const int N = T::DIMENSION;
	static SynchronizedBuffer<int> min_max_buf(2 * N);

	for (int i = 0; i < N; i++)
	{
		min_max_buf[i] = floatToOrderedInt(FLT_MAX);
		min_max_buf[N + i] = floatToOrderedInt(-FLT_MAX);
	}
	min_max_buf.setOnCPU();
	min_max_buf.Synchronize();

	Launch(img, buffer, min_max_buf.getDevicePtr(), generic_normalize_float_op<N, T>());
	min_max_buf.Synchronize();

	for (int i = 0; i < N; i++)
	{
		min_res[i] = orderedIntToFloat(min_max_buf[i]);
		max_res[i] = orderedIntToFloat(min_max_buf[N + i]);
	}
}

template<> inline void compute_range(Image& img, PixelDebugVisualizer<float>& buffer, float& min_res, float& max_res)
{
	static SynchronizedBuffer<int> min_max_buf(2);

	min_max_buf[0] = floatToOrderedInt(FLT_MAX);
	min_max_buf[1] = floatToOrderedInt(-FLT_MAX);
	min_max_buf.setOnCPU();
	min_max_buf.Synchronize();

	Launch(img, buffer, min_max_buf.getDevicePtr(), generic_normalize_float_op<1, float>());
	min_max_buf.Synchronize();

	min_res = orderedIntToFloat(min_max_buf[0]);
	max_res = orderedIntToFloat(min_max_buf[1]);
}

template<typename T> struct normalized_data
{
	bool normalize;
	T min, max;
};
template<typename T> normalized_data<T> generate_normalize_data(PixelDebugVisualizer<T>& buffer, Image& img)
{
	normalized_data<T> arg;
	arg.normalize = buffer.m_normalizationData.type != PixelDebugVisualizerBase<T>::NormalizationType::None;
	if (buffer.m_normalizationData.type == PixelDebugVisualizerBase<T>::NormalizationType::Adaptive)
	{
		compute_range(img, buffer, arg.min, arg.max);
	}
	else
	{
		arg.min = buffer.m_normalizationData.min;
		arg.max = buffer.m_normalizationData.max;
	}

	return arg;
}

}
