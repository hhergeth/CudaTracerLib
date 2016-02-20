#pragma once

#include "MathFunc.h"
#include "Vector.h"

//Implementation copied from Mitsuba.

namespace CudaTracerLib {

class Spline
{
public:
	CTL_EXPORT CUDA_DEVICE CUDA_HOST static float evalCubicInterp1D(float x, const float *values, size_t size, float min, float max, bool extrapolate = false);
	CTL_EXPORT CUDA_DEVICE CUDA_HOST static float evalCubicInterp1DN(float x, const float *nodes, const float *values, size_t size, bool extrapolate = false);
	CTL_EXPORT CUDA_DEVICE CUDA_HOST static float integrateCubicInterp1D(size_t idx, const float *values, size_t size, float min, float max);
	CTL_EXPORT CUDA_DEVICE CUDA_HOST static float integrateCubicInterp1DN(size_t idx, const float *nodes, const float *values, size_t size);
	CTL_EXPORT CUDA_DEVICE CUDA_HOST static float sampleCubicInterp1D(size_t idx, float *values, size_t size, float min, float max, float sample, float *fval = NULL);
	CTL_EXPORT CUDA_DEVICE CUDA_HOST static float sampleCubicInterp1DN(size_t idx, float *nodes, float *values, size_t size, float sample, float *fval = NULL);
	CTL_EXPORT CUDA_DEVICE CUDA_HOST static float evalCubicInterp2D(const Vec2f &p, const float *values, const uint2 &size, const Vec2f &min, const Vec2f &max, bool extrapolate = false);
	CTL_EXPORT CUDA_DEVICE CUDA_HOST static float evalCubicInterp2DN(const Vec2f &p, const float **nodes_, const float *values, const uint2 &size, bool extrapolate = false);
	CTL_EXPORT CUDA_DEVICE CUDA_HOST static float evalCubicInterp3D(const Vec3f &p, const float *values, const uint3 &size, const Vec3f &min, const Vec3f &max, bool extrapolate = false);
	CTL_EXPORT CUDA_DEVICE CUDA_HOST static float evalCubicInterp3DN(const Vec3f &p, const float **nodes_, const float *values, const uint3 &size, bool extrapolate = false);
};

}