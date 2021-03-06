#include "ShapeSet.h"
#include <Math/AlgebraHelper.h>
#include "TriIntersectorData.h"
#include "TriangleData.h"
#include <Kernel/TraceHelper.h>
#include <Math/Warp.h>
#include <Math/MonteCarlo.h>

namespace CudaTracerLib {

void ShapeSet::triData::Recalculate(const float4x4& mat, const TriIntersectorData& T, const TriangleData& TData)
{
	T.getData(p[0], p[1], p[2]);
	auto v1 = mat.TransformDirection(p[0] - p[1]), v2 = mat.TransformDirection(p[2] - p[1]);
	DifferentialGeometry dg;
	dg.bary = Vec2f(1.0f / 3.0f);
	TData.fillDG(mat, dg);
	n = dg.sys.n;
	for (unsigned int i = 0; i < 3; i++)
		p[i] = mat.TransformPoint(p[i]);
	area = 0.5f * length(cross(p[2] - p[0], p[1] - p[0]));
}

CUDA_FUNC_IN void getUV(TriangleData& dat, const Vec2f& bary, Vec2f& uv)
{
	Vec2f a, b, c;
	dat.getUVSetData(0, a, b, c);
	float u = bary.x, v = bary.y, w = 1 - u - v;
	uv = u * a + v * b + w * c;
}

template<typename T> CUDA_FUNC_IN T* getData(unsigned int idx)
{
    return (T*)&g_SceneData.m_sAnimData[idx];
}

unsigned int ShapeSet::sampleTriangle(Vec3f& p0, Vec3f& p1, Vec3f& p2, Vec2f& uv0, Vec2f& uv1, Vec2f& uv2, float& pdf, float sample) const
{
    float* areaDistribution = getData<float>(m_areaDistributionIndex);
    triData* triangles = getData<triData>(m_trianglesIndex);

	unsigned int index = MonteCarlo::sampleReuse(areaDistribution, count, sample, pdf);
	const triData& sn = triangles[index];
	g_SceneData.m_sTriData[sn.tDat].getUVSetData(0, uv0, uv1, uv2);
	p0 = sn.p[0];
	p1 = sn.p[1];
	p2 = sn.p[2];
	return index;
}

void ShapeSet::SamplePosition(PositionSamplingRecord& pRec, const Vec2f& spatialSample, Vec2f* uv) const
{
    float* areaDistribution = getData<float>(m_areaDistributionIndex);
    triData* triangles = getData<triData>(m_trianglesIndex);

	float pdf;
	Vec2f sample = spatialSample;
	unsigned int index = MonteCarlo::sampleReuse(areaDistribution, count, sample.y, pdf);
	const triData& sn = triangles[index];
	Vec2f bary = Warp::squareToUniformTriangle(sample);
	pRec.p = bary.x * sn.p[0] + bary.y * sn.p[1] + (1.f - bary.x - bary.y) * sn.p[2];
	//pRec.n = normalize(cross(sn.p[1] - sn.p[0], sn.p[2] - sn.p[0]));
	pRec.n = sn.n;
	pRec.pdf = 1.0f / sumArea;
	pRec.measure = EArea;
	pRec.uv = bary;
	if (uv)
		getUV(g_SceneData.m_sTriData[sn.tDat], bary, *uv);
}

bool ShapeSet::getPosition(const Vec3f& pos, Vec2f* bary, Vec2f* uv) const
{
    float* areaDistribution = getData<float>(m_areaDistributionIndex);
    triData* triangles = getData<triData>(m_trianglesIndex);

	for (unsigned int i = 0; i < count; i++)
	{
		const triData& sn = triangles[i];
		Vec2f b;
		if (AlgebraHelper::Barycentric(pos, sn.p[0], sn.p[1], sn.p[2], b.x, b.y))
		{
			if (bary)
				*bary = b;
			if (uv)
				getUV(g_SceneData.m_sTriData[sn.tDat], b, *uv);
			return true;
		}
	}
	return false;
}

float ShapeSet::PdfTriangle(const Vec3f& pos) const
{
    float* areaDistribution = getData<float>(m_areaDistributionIndex);
    triData* triangles = getData<triData>(m_trianglesIndex);

	for (unsigned int i = 0; i < count; i++)
	{
		const triData& sn = triangles[i];
		Vec2f b;
		if (AlgebraHelper::Barycentric(pos, sn.p[0], sn.p[1], sn.p[2], b.x, b.y))
			return areaDistribution[i + 1] - areaDistribution[i];
	}
	return 0.0f;
}

AABB ShapeSet::getBox() const
{
    float* areaDistribution = getData<float>(m_areaDistributionIndex);
    triData* triangles = getData<triData>(m_trianglesIndex);

    AABB b = AABB::Identity();
    for (unsigned int i = 0; i < count; i++)
        b = b.Extend(triangles[i].box());
    return b;
}

}