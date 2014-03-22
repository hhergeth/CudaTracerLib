#pragma once

#include <MathTypes.h>

struct e_TriangleData;
class e_Node;
struct e_KernelBSDF;
struct e_KernelMaterial;
struct e_KernelBSSRDF;
struct BSDFSamplingRecord;
struct e_TriIntersectorData;
struct TraceResult
{
	float m_fDist;
	float2 m_fUV;
	const e_TriangleData* m_pTri;
	const e_Node* m_pNode;
	const e_TriIntersectorData* m_pInt;
	CUDA_DEVICE CUDA_HOST bool hasHit() const;
	CUDA_DEVICE CUDA_HOST void Init();
	CUDA_DEVICE CUDA_HOST operator bool() const;
	CUDA_DEVICE CUDA_HOST void lerpFrame(Frame& sys) const;
	CUDA_DEVICE CUDA_HOST unsigned int getMatIndex() const;
	CUDA_DEVICE CUDA_HOST float2 lerpUV() const;
	CUDA_DEVICE CUDA_HOST Spectrum Le(const float3& p, const Frame& sys, const float3& w) const;
	CUDA_DEVICE CUDA_HOST unsigned int LightIndex() const;
	CUDA_DEVICE CUDA_HOST const e_KernelMaterial& getMat() const;
	CUDA_DEVICE CUDA_HOST void getBsdfSample(const Ray& r, CudaRNG& _rng, BSDFSamplingRecord* bRec) const;
	CUDA_DEVICE CUDA_HOST void getBsdfSample(const Ray& r, CudaRNG& _rng, BSDFSamplingRecord* bRec, const float3& wo) const;
};