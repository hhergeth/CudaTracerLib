#pragma once

#include <MathTypes.h>

struct e_TriangleData;
class e_Node;
struct e_KernelBSDF;
struct e_KernelMaterial;
struct e_KernelBSSRDF;
struct BSDFSamplingRecord;
struct e_TriIntersectorData;
struct DifferentialGeometry;
struct TraceResult
{
	float m_fDist;
	Vec2f m_fUV;
	const e_TriangleData* m_pTri;
	const e_Node* m_pNode;
	CUDA_DEVICE CUDA_HOST bool hasHit() const;
	CUDA_DEVICE CUDA_HOST void Init();
	CUDA_DEVICE CUDA_HOST operator bool() const;
	CUDA_DEVICE CUDA_HOST void fillDG(DifferentialGeometry& dg) const;
	CUDA_DEVICE CUDA_HOST unsigned int getMatIndex() const;
	CUDA_DEVICE CUDA_HOST Spectrum Le(const Vec3f& p, const Frame& sys, const Vec3f& w) const;
	CUDA_DEVICE CUDA_HOST unsigned int LightIndex() const;
	CUDA_DEVICE CUDA_HOST const e_KernelMaterial& getMat() const;
	CUDA_DEVICE CUDA_HOST void getBsdfSample(const Ray& r, CudaRNG& _rng, BSDFSamplingRecord* bRec) const;
	CUDA_DEVICE CUDA_HOST void getBsdfSample(const Ray& r, CudaRNG& _rng, BSDFSamplingRecord* bRec, const Vec3f& wo) const;
	//wi points towards p
	CUDA_DEVICE CUDA_HOST void getBsdfSample(const Vec3f& wi, const Vec3f& p, BSDFSamplingRecord* bRec, CudaRNG* _rng = 0) const;
	CUDA_DEVICE CUDA_HOST unsigned int getNodeIndex() const;
};