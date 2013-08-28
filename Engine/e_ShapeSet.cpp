#include <StdAfx.h>
#include "e_ShapeSet.h"

AABB ShapeSet::triData::box() const
{
	AABB b = AABB::Identity();
	for(int i = 0; i < 3; i++)
		b.Enlarge(p[i]);
	return b;
}

void ShapeSet::triData::Recalculate(float4x4& mat)
{
	datRef->getData(p[0], p[1], p[2]);
	for(int i = 0; i < 3; i++)
		p[i] = mat * p[i];
	dat.setData(p[0], p[1], p[2]);
	n = -cross(p[2] - p[0], p[1] - p[0]);
	area = 0.5f * length(n);
	n = normalize(n);
}