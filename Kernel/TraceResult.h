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
	unsigned int m_triIdx;
	unsigned int m_nodeIdx;
	CUDA_FUNC_IN bool hasHit() const
	{
		return m_triIdx != UINT_MAX;
	}
	CUDA_FUNC_IN void Init()
	{
		m_fDist = FLT_MAX;
		m_triIdx = UINT_MAX;
		m_nodeIdx = UINT_MAX;
	}
	CUDA_DEVICE CUDA_HOST unsigned int getMatIndex() const;
	CUDA_DEVICE CUDA_HOST Spectrum Le(const Vec3f& p, const Frame& sys, const NormalizedT<Vec3f>& w) const;
	CUDA_DEVICE CUDA_HOST unsigned int LightIndex() const;
	CUDA_DEVICE CUDA_HOST const Material& getMat() const;
	CUDA_DEVICE CUDA_HOST unsigned int getNodeIndex() const;
	CUDA_DEVICE CUDA_HOST unsigned int getTriIndex() const;

	CUDA_DEVICE CUDA_HOST void fillDG(DifferentialGeometry& dg) const;
	//wi towards p, wo away
	CUDA_DEVICE CUDA_HOST void getBsdfSample(const NormalizedT<Ray>& r, BSDFSamplingRecord& bRec, ETransportMode mode, CudaRNG* rng, const Spectrum* f_i = 0, const NormalizedT<Vec3f>* wo = 0) const;
	//wi towards p, wo away
	CUDA_DEVICE CUDA_HOST void getBsdfSample(const NormalizedT<Vec3f>& wi, const Vec3f& p, BSDFSamplingRecord& bRec, ETransportMode mode, CudaRNG* rng, const Spectrum* f_i = 0, const NormalizedT<Vec3f>* wo = 0) const;
};

}