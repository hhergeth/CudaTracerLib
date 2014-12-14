#include "e_TraceResult.h"
#include "e_TriangleData.h"
#include "e_Node.h"

bool TraceResult::hasHit() const
{
	return m_pTri != 0;
}

TraceResult::operator bool() const
{
	return hasHit();
}

void TraceResult::Init()
{
	m_fDist = FLT_MAX;
	m_pNode = 0;
	m_pTri = 0;
	m_pInt = 0;
}

unsigned int TraceResult::getMatIndex() const
{
	return m_pTri->getMatIndex(m_pNode->m_uMaterialOffset);
}

void TraceResult::getBsdfSample(const Ray& r, CudaRNG& _rng, BSDFSamplingRecord* bRec, const float3& wo) const
{
	getBsdfSample(r, _rng, bRec);
	bRec->wo = bRec->dg.toLocal(wo);
}

void TraceResult::getBsdfSample(const Ray& r, CudaRNG& _rng, BSDFSamplingRecord* bRec) const
{
	getBsdfSample(r.direction, r(m_fDist), bRec, &_rng);
}