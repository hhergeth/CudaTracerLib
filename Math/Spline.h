#pragma once

#include "cutil_math.h"

class Spline
{
public:
	CUDA_DEVICE CUDA_HOST static float evalCubicInterp1D(float x, const float *values, size_t size, float min, float max, bool extrapolate = false);
	CUDA_DEVICE CUDA_HOST static float evalCubicInterp1DN(float x, const float *nodes, const float *values, size_t size, bool extrapolate = false);
	CUDA_DEVICE CUDA_HOST static float integrateCubicInterp1D(size_t idx, const float *values, size_t size, float min, float max);
	CUDA_DEVICE CUDA_HOST static float integrateCubicInterp1DN(size_t idx, const float *nodes, const float *values, size_t size);
	CUDA_DEVICE CUDA_HOST static float sampleCubicInterp1D(size_t idx, float *values, size_t size, float min, float max, float sample, float *fval = NULL);
	CUDA_DEVICE CUDA_HOST static float sampleCubicInterp1DN(size_t idx, float *nodes, float *values, size_t size, float sample, float *fval = NULL);
	CUDA_DEVICE CUDA_HOST static float evalCubicInterp2D(const float2 &p, const float *values, const uint2 &size, const float2 &min, const float2 &max, bool extrapolate = false);
	CUDA_DEVICE CUDA_HOST static float evalCubicInterp2DN(const float2 &p, const float **nodes_, const float *values, const uint2 &size, bool extrapolate = false);
	CUDA_DEVICE CUDA_HOST static float evalCubicInterp3D(const float3 &p, const float *values, const uint3 &size, const float3 &min, const float3 &max, bool extrapolate = false);
	CUDA_DEVICE CUDA_HOST static float evalCubicInterp3DN(const float3 &p, const float **nodes_, const float *values, const uint3 &size, bool extrapolate = false);
};