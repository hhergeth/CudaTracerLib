#pragma once

#include <Math/Vector.h>
#include "Sampler_device.h"
#include <SceneTypes/Samples.h>

namespace CudaTracerLib {

struct TraceResult;
struct Material;
struct Spectrum;
struct Ray;

CUDA_FUNC_IN float PdfWtoA(const float aPdfW, const float aDist2, const float aCosThere)
{
	return aPdfW * math::abs(aCosThere) / aDist2;
}

CUDA_FUNC_IN float PdfAtoW(const float aPdfA, const float aDist2, const float aCosThere)
{
	return aPdfA * aDist2 / math::abs(aCosThere);
}

CTL_EXPORT CUDA_HOST CUDA_DEVICE DirectSamplingRecord DirectSamplingRecFromRay(const NormalizedT<Ray>& r, float dist, const NormalizedT<Vec3f>& last_nor, const Vec3f& P, const NormalizedT<Vec3f>& n, EMeasure measure = ESolidAngle);

CTL_EXPORT CUDA_HOST CUDA_DEVICE bool V(const Vec3f& a, const Vec3f& b, TraceResult* res = 0);

CTL_EXPORT CUDA_HOST CUDA_DEVICE float G(const NormalizedT<Vec3f>& N_x, const NormalizedT<Vec3f>& N_y, const Vec3f& x, const Vec3f& y);

CTL_EXPORT CUDA_HOST CUDA_DEVICE Spectrum Transmittance(const Ray& r, float tmin, float tmax);

CTL_EXPORT CUDA_HOST CUDA_DEVICE Spectrum UniformSampleAllLights(const BSDFSamplingRecord& bRec, const Material& mat, int nSamples, Sampler& rng, bool attenuated = false, bool use_mis = true);

CTL_EXPORT CUDA_HOST CUDA_DEVICE Spectrum UniformSampleOneLight(const BSDFSamplingRecord& bRec, const Material& mat, Sampler& rng, bool attenuated = false, bool use_mis = true);

}