#include <StdAfx.h>
#include "e_ShapeSet.h"

AABB ShapeSet::triData::box() const
{
	AABB b = AABB::Identity();
	for(int i = 0; i < 3; i++)
		b.Enlarge(p[i]);
	return b;
}

void ShapeSet::triData::Recalculate(const float4x4& mat)
{
	datRef->getData(p[0], p[1], p[2]);
	for(int i = 0; i < 3; i++)
		p[i] = mat.TransformPoint(p[i]);
	n = -cross(p[2] - p[0], p[1] - p[0]);
	area = 0.5f * length(n);
	n = normalize(n);
}

ShapeSet::ShapeSet(e_StreamReference(e_TriIntersectorData)* indices, unsigned int indexCount, float4x4& mat)
{
	if(indexCount > max_SHAPE_LENGTH)
		throw 1;
	count = indexCount;
	float areas[max_SHAPE_LENGTH];
	sumArea = 0;
	for(int i = 0; i < count; i++)
	{
		tris[i] = triData(indices[i], mat);
		areas[i] = tris[i].area;
		sumArea += tris[i].area;
	}
	areaDistribution = Distribution1D<max_SHAPE_LENGTH>(areas, count);
	areaDistribution.Normalize();
}

void ShapeSet::Recalculate(const float4x4& mat)
{
	float areas[max_SHAPE_LENGTH];
	sumArea = 0;
	for(int i = 0; i < count; i++)
	{
		tris[i].Recalculate(mat);
		areas[i] = tris[i].area;
		sumArea += tris[i].area;
	}
	areaDistribution = Distribution1D<max_SHAPE_LENGTH>(areas, count);
	areaDistribution.Normalize();
}