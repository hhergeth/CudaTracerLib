#include "e_ShapeSet.h"

float3 ShapeSet::triData::Sample(float u1, float u2, float3* Ns, float* a, float* b) const
{
	float b1, b2;
	UniformSampleTriangle(u1, u2, &b1, &b2);
	*Ns = nor();
	if(a)
		*a = b1;
	if(b)
		*b = b2;
	return rndPoint(b1, b2);
}

void ShapeSet::SamplePosition(PositionSamplingRecord& pRec, const float2& spatialSample) const
{
	float pdf;
	float2 sample = spatialSample;
	unsigned int index = areaDistribution.SampleReuse(sample.y, pdf);
	const triData& sn = tris[index];
	float2 bary = Warp::squareToUniformTriangle(sample);
	pRec.p = bary.x * sn.p[0] + bary.y * sn.p[1] + (1.f - bary.x - bary.y) * sn.p[2];
	pRec.n = sn.n;
	pRec.pdf = 1.0f / sumArea;
	pRec.measure = EArea;
	pRec.uv = bary;
}