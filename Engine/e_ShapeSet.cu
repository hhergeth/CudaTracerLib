#include "e_ShapeSet.h"

//unsigned int index = min(unsigned int(size - 2U), max(0U, unsigned int(entry - cdf - 1)));
//const float *ptr = STL_lower_bound(cdf, cdf + size, sample);
//unsigned int index = (unsigned int)min(int(size - 2U), max(0, int(ptr - cdf) - 1));
unsigned int sampleReuse(float *cdf, unsigned int size, float &sample, float& pdf)
{
	const float *entry = STL_lower_bound(cdf, cdf + size + 1, sample);
	unsigned int index = (unsigned int )min(int(entry - cdf) - 1, int(size - 1));
	pdf = cdf[index + 1] - cdf[index];
	sample = (sample - cdf[index]) / pdf;
	return index;
}

void ShapeSet::SamplePosition(PositionSamplingRecord& pRec, const Vec2f& spatialSample) const
{
	float pdf;
	Vec2f sample = spatialSample;
	unsigned int index = sampleReuse(areaDistribution.operator*(), count, sample.y, pdf);
	const triData& sn = triangles[index];
	Vec2f bary = Warp::squareToUniformTriangle(sample);
	pRec.p = bary.x * sn.p[0] + bary.y * sn.p[1] + (1.f - bary.x - bary.y) * sn.p[2];
	pRec.n = -normalize(cross(sn.p[2] - sn.p[0], sn.p[1] - sn.p[0]));
	pRec.pdf = 1.0f / sumArea;
	pRec.measure = EArea;
	pRec.uv = bary;
}