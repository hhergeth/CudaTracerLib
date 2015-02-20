#pragma once

#include <MathTypes.h>
#include "../Engine/e_Node.h"
#include "../Engine/e_TriangleData.h"

struct BSDFSamplingRecord;
struct DifferentialGeometry;
struct TraceResult
{
	float m_fDist;
	Vec2f m_fBaryCoords;
	const e_TriangleData* m_pTri;
	const e_Node* m_pNode;
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
	CUDA_FUNC_IN unsigned int getMatIndex() const
	{
		return m_pTri->getMatIndex(m_pNode->m_uMaterialOffset);
	}
	CUDA_DEVICE CUDA_HOST Spectrum Le(const Vec3f& p, const Frame& sys, const Vec3f& w) const;
	CUDA_DEVICE CUDA_HOST unsigned int LightIndex() const;
	CUDA_DEVICE CUDA_HOST const e_KernelMaterial& getMat() const;
	CUDA_DEVICE CUDA_HOST unsigned int getNodeIndex() const;
	CUDA_DEVICE CUDA_HOST unsigned int getTriIndex() const;

	CUDA_DEVICE CUDA_HOST void fillDG(DifferentialGeometry& dg) const;
	//wi towards p, wo away
	CUDA_DEVICE CUDA_HOST void getBsdfSample(const Ray& r, BSDFSamplingRecord& bRec, ETransportMode mode, const Vec3f* wo = 0) const;
	//wi towards p, wo away
	CUDA_DEVICE CUDA_HOST void getBsdfSample(const Vec3f& wi, const Vec3f& p, BSDFSamplingRecord& bRec, ETransportMode mode, const Vec3f* wo = 0) const;
};