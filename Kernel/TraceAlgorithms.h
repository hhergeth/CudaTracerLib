#pragma once

#include <Math/Vector.h>
#include "Sampler_device.h"

namespace CudaTracerLib {

struct TraceResult;
struct BSDFSamplingRecord;
struct Material;
struct Spectrum;
struct Ray;

CTL_EXPORT CUDA_HOST CUDA_DEVICE bool V(const Vec3f& a, const Vec3f& b, TraceResult* res = 0);

CTL_EXPORT CUDA_HOST CUDA_DEVICE float G(const NormalizedT<Vec3f>& N_x, const NormalizedT<Vec3f>& N_y, const Vec3f& x, const Vec3f& y);

CTL_EXPORT CUDA_HOST CUDA_DEVICE Spectrum Transmittance(const Ray& r, float tmin, float tmax);

CTL_EXPORT CUDA_HOST CUDA_DEVICE Spectrum UniformSampleAllLights(const BSDFSamplingRecord& bRec, const Material& mat, int nSamples, Sampler& rng, bool attenuated = false);

CTL_EXPORT CUDA_HOST CUDA_DEVICE Spectrum UniformSampleOneLight(const BSDFSamplingRecord& bRec, const Material& mat, Sampler& rng, bool attenuated = false);

}