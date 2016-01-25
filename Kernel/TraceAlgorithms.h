#pragma once

#include <Math/Vector.h>

namespace CudaTracerLib {

struct TraceResult;
struct BSDFSamplingRecord;
struct Material;
struct CudaRNG;
struct Spectrum;
struct Ray;

CUDA_HOST CUDA_DEVICE bool V(const Vec3f& a, const Vec3f& b, TraceResult* res = 0);

CUDA_HOST CUDA_DEVICE float G(const NormalizedT<Vec3f>& N_x, const NormalizedT<Vec3f>& N_y, const Vec3f& x, const Vec3f& y);

CUDA_HOST CUDA_DEVICE Spectrum Transmittance(const Ray& r, float tmin, float tmax);

CUDA_HOST CUDA_DEVICE Spectrum UniformSampleAllLights(const BSDFSamplingRecord& bRec, const Material& mat, int nSamples, CudaRNG& rng, bool attenuated = false);

CUDA_HOST CUDA_DEVICE Spectrum UniformSampleOneLight(const BSDFSamplingRecord& bRec, const Material& mat, CudaRNG& rng, bool attenuated = false);

}