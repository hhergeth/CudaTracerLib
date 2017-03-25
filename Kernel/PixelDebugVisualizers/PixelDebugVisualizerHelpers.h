#pragma once
#include <Math/Vector.h>
#include <Math/Spectrum.h>
#include <Math/MathFunc.h>

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

}
