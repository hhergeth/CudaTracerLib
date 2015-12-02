#pragma once

#include <MathTypes.h>

namespace CudaTracerLib {

struct BSDFSamplingRecord;
struct DifferentialGeometry;
class Node;
struct TriangleData;
struct Material;
struct BSDFSamplingRecord;
enum ETransportMode : int;
struct CudaRNG;

struct TraceResult
{
	float m_fDist;
	Vec2f m_fBaryCoords;
	const TriangleData* m_pTri;
	const Node* m_pNode;
	CUDA_FUNC_IN bool hasHit() const
	{
		return m_pTri != 0;
	}
	CUDA_FUNC_IN void Init()
	{
		m_fDist = FLT_MAX;
		m_pNode = 0;
		m_pTri = 0;
	}
	CUDA_DEVICE CUDA_HOST unsigned int getMatIndex() const;
	CUDA_DEVICE CUDA_HOST Spectrum Le(const Vec3f& p, const Frame& sys, const Vec3f& w) const;
	CUDA_DEVICE CUDA_HOST unsigned int LightIndex() const;
	CUDA_DEVICE CUDA_HOST const Material& getMat() const;
	CUDA_DEVICE CUDA_HOST unsigned int getNodeIndex() const;
	CUDA_DEVICE CUDA_HOST unsigned int getTriIndex() const;

	CUDA_DEVICE CUDA_HOST void fillDG(DifferentialGeometry& dg) const;
	//wi towards p, wo away
	CUDA_DEVICE CUDA_HOST void getBsdfSample(const Ray& r, BSDFSamplingRecord& bRec, ETransportMode mode, CudaRNG* rng, const Vec3f* wo = 0) const;
	//wi towards p, wo away
	CUDA_DEVICE CUDA_HOST void getBsdfSample(const Vec3f& wi, const Vec3f& p, BSDFSamplingRecord& bRec, ETransportMode mode, CudaRNG* rng, const Vec3f* wo = 0) const;
};

}