#include <StdAfx.h>
#include "e_ShapeSet.h"

AABB ShapeSet::triData::box() const
{
	AABB b = AABB::Identity();
	for (unsigned int i = 0; i < 3; i++)
		b.Enlarge(p[i]);
	return b;
}

void ShapeSet::triData::Recalculate(const float4x4& mat)
{
	iDat->getData(p[0], p[1], p[2]);
	for (unsigned int i = 0; i < 3; i++)
		p[i] = mat.TransformPoint(p[i]);
	Vec3f n = -cross(p[2] - p[0], p[1] - p[0]);
	area = 0.5f * length(n);
}

ShapeSet::ShapeSet(e_StreamReference(e_TriIntersectorData)* indices, unsigned int indexCount, float4x4& mat, e_Stream<char>* buffer)
{
	count = indexCount;
	buffer1 = buffer->malloc((count + 1) * sizeof(float));
	buffer2 = buffer->malloc(count * sizeof(triData));
	areaDistribution = buffer1.AsVar<float>();
	triangles = buffer2.AsVar<triData>();
	for (unsigned int i = 0; i < count; i++)
		triangles[i].iDat = indices[i].AsVar();
	Recalculate(mat);
}

void ShapeSet::Recalculate(const float4x4& mat)
{
	buffer1.Invalidate();
	buffer2.Invalidate();
	sumArea = 0;
	areaDistribution[0] = 0.0f;
	for (unsigned int i = 0; i < count; i++)
	{
		triangles[i].Recalculate(mat);
		sumArea += triangles[i].area;
		areaDistribution[i + 1] = areaDistribution[i] + triangles[i].area;
	}
	//pdf values have to be converted to normalized cdf
	for (unsigned int i = 0; i <= count; i++)
		areaDistribution[i] = areaDistribution[i] / sumArea;
}