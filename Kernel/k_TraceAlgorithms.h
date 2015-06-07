#pragma once

#include "../MathTypes.h"

struct TraceResult;
struct BSDFSamplingRecord;
struct e_KernelMaterial;
struct CudaRNG;

CUDA_HOST CUDA_DEVICE bool V(const Vec3f& a, const Vec3f& b, TraceResult* res = 0);

CUDA_HOST CUDA_DEVICE float G(const Vec3f& N_x, const Vec3f& N_y, const Vec3f& x, const Vec3f& y);

CUDA_HOST CUDA_DEVICE Spectrum Transmittance(const Ray& r, float tmin, float tmax, unsigned int a_NodeIndex = 0xffffffff);

CUDA_HOST CUDA_DEVICE Spectrum UniformSampleAllLights(const BSDFSamplingRecord& bRec, const e_KernelMaterial& mat, int nSamples, CudaRNG& rng);

CUDA_HOST CUDA_DEVICE Spectrum UniformSampleOneLight(const BSDFSamplingRecord& bRec, const e_KernelMaterial& mat, CudaRNG& rng);