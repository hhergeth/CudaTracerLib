#include "TraceResult.h"
#include <Engine/TriangleData.h>
#include <Engine/Node.h>
#include "TraceHelper.h"
#include <Engine/Light.h>
#include <Engine/Samples.h>

namespace CudaTracerLib {

void TraceResult::getBsdfSample(const Ray& r, BSDFSamplingRecord& bRec, ETransportMode mode, CudaRNG* rng, const Spectrum* f_i, const NormalizedT<Vec3f>* wo) const
{
	getBsdfSample(r.dirUnit(), r(m_fDist), bRec, mode, rng, f_i, wo);
}

void TraceResult::getBsdfSample(const NormalizedT<Vec3f>& wi, const Vec3f& p, BSDFSamplingRecord& bRec, ETransportMode mode, CudaRNG* rng, const Spectrum* f_i, const NormalizedT<Vec3f>* wo) const
{
	if (f_i)
		bRec.f_i = *f_i;
	else
	{
		bRec.f_i = Spectrum(0.0f);
		if (mode == ETransportMode::EImportance)
			printf("Please set the incoming radiance f_i when doing photon tracing!\n");
	}
	bRec.rng = rng;
	bRec.eta = 1.0f;
	bRec.mode = mode;
	bRec.sampledType = 0;
	bRec.typeMask = ETypeCombinations::EAll;
	bRec.dg.P = p;
	fillDG(bRec.dg);
	bRec.wi = bRec.dg.toLocal(-wi);
	if (wo)
		bRec.wo = bRec.dg.toLocal(*wo);
	getMat().SampleNormalMap(bRec.dg, wi * m_fDist);
}

Spectrum TraceResult::Le(const Vec3f& p, const Frame& sys, const NormalizedT<Vec3f>& w) const
{
	unsigned int i = LightIndex();
	if (i == UINT_MAX)
		return Spectrum(0.0f);
	else return g_SceneData.getLight(*this)->eval(p, sys, w);
}

unsigned int TraceResult::LightIndex() const
{
	unsigned int nli = getMat().NodeLightIndex;
	if (nli == UINT_MAX)
		return UINT_MAX;
	return g_SceneData.m_sNodeData[m_nodeIdx].m_uLights(nli);
}

unsigned int TraceResult::getNodeIndex() const
{
	return m_nodeIdx;
}

const Material& TraceResult::getMat() const
{
	return g_SceneData.m_sMatData[getMatIndex()];
}

unsigned int TraceResult::getTriIndex() const
{
	return m_triIdx;
}

void TraceResult::fillDG(DifferentialGeometry& dg) const
{
	CudaTracerLib::fillDG(m_fBaryCoords, &g_SceneData.m_sTriData[m_triIdx], &g_SceneData.m_sNodeData[m_nodeIdx], dg);
}

unsigned int TraceResult::getMatIndex() const
{
	return g_SceneData.m_sTriData[m_triIdx].getMatIndex(g_SceneData.m_sNodeData[m_nodeIdx].m_uMaterialOffset);
}

}
