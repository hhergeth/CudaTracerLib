#include "ShapeSet.h"
#include <Math/Sampling.h>
#include "Samples.h"
#include "TriIntersectorData.h"
#include "TriangleData.h"

namespace CudaTracerLib {

CUDA_FUNC_IN void getUV(TriangleData* dat, const Vec2f& bary, Vec2f& uv)
{
	Vec2f a, b, c;
	dat->getUVSetData(0, a, b, c);
	float u = bary.x, v = bary.y, w = 1 - u - v;
	uv = u * a + v * b + w * c;
}

void ShapeSet::SamplePosition(PositionSamplingRecord& pRec, const Vec2f& spatialSample, Vec2f* uv) const
{
	float pdf;
	Vec2f sample = spatialSample;
	unsigned int index = MonteCarlo::sampleReuse(areaDistribution.operator*(), count, sample.y, pdf);
	const triData& sn = triangles[index];
	Vec2f bary = Warp::squareToUniformTriangle(sample);
	pRec.p = bary.x * sn.p[0] + bary.y * sn.p[1] + (1.f - bary.x - bary.y) * sn.p[2];
	pRec.n = -normalize(cross(sn.p[2] - sn.p[0], sn.p[1] - sn.p[0]));
	pRec.pdf = 1.0f / sumArea;
	pRec.measure = EArea;
	pRec.uv = bary;
	if (uv)
		getUV(sn.tDat.operator*(), bary, *uv);
}

bool ShapeSet::getPosition(const Vec3f& pos, Vec2f* bary, Vec2f* uv) const
{
	for (unsigned int i = 0; i < count; i++)
	{
		const triData& sn = triangles[i];
		Vec2f b;
		if (MonteCarlo::Barycentric(pos, sn.p[0], sn.p[1], sn.p[2], b.x, b.y))
		{
			if (bary)
				*bary = b;
			if (uv)
				getUV(sn.tDat.operator*(), b, *uv);
			return true;
		}
	}
	return false;
}

}