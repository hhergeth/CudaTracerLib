#include "k_TraceResult.h"
#include <Engine/e_TriangleData.h>
#include <Engine/e_Node.h>
#include "k_TraceHelper.h"
#include <Engine/e_Light.h>
#include <Engine/e_Samples.h>

namespace CudaTracerLib {

void TraceResult::getBsdfSample(const Ray& r, BSDFSamplingRecord& bRec, ETransportMode mode, CudaRNG* rng, const Vec3f* wo) const
{
	getBsdfSample(r.direction, r(m_fDist), bRec, mode, rng, wo);
}

void TraceResult::getBsdfSample(const Vec3f& wi, const Vec3f& p, BSDFSamplingRecord& bRec, ETransportMode mode, CudaRNG* rng, const Vec3f* wo) const
{
	bRec.rng = rng;
	bRec.eta = 1.0f;
	bRec.mode = mode;
	bRec.sampledType = 0;
	bRec.typeMask = ETypeCombinations::EAll;
	bRec.dg.P = p;
	fillDG(bRec.dg);
	bRec.wi = normalize(bRec.dg.toLocal(-wi));
	if (wo)
		bRec.wo = normalize(bRec.dg.toLocal(*wo));
	getMat().SampleNormalMap(bRec.dg, wi * m_fDist);
}

Spectrum TraceResult::Le(const Vec3f& p, const Frame& sys, const Vec3f& w) const
{
	unsigned int i = LightIndex();
	if (i == UINT_MAX)
		return Spectrum(0.0f);
	else return g_SceneData.m_sLightData[i].eval(p, sys, w);
}

unsigned int TraceResult::LightIndex() const
{
	unsigned int nli = getMat().NodeLightIndex;
	if (nli == UINT_MAX)
		return UINT_MAX;
	return m_pNode->m_uLights(nli);
}

unsigned int TraceResult::getNodeIndex() const
{
	return m_pNode - g_SceneData.m_sNodeData.Data;
}

const e_KernelMaterial& TraceResult::getMat() const
{
	return g_SceneData.m_sMatData[getMatIndex()];
}

unsigned int TraceResult::getTriIndex() const
{
	return m_pTri - g_SceneData.m_sTriData.Data;
}

void TraceResult::fillDG(DifferentialGeometry& dg) const
{
	CudaTracerLib::fillDG(m_fBaryCoords, m_pTri, m_pNode, dg);
}

unsigned int TraceResult::getMatIndex() const
{
	return m_pTri->getMatIndex(m_pNode->m_uMaterialOffset);
}

}