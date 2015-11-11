#include "ShapeSet.h"
#include <Math/Sampling.h>
#include "Samples.h"
#include "TriIntersectorData.h"

namespace CudaTracerLib {

void ShapeSet::SamplePosition(PositionSamplingRecord& pRec, const Vec2f& spatialSample) const
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
}

}