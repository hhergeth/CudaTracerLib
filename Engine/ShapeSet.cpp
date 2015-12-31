#include <StdAfx.h>
#include "ShapeSet.h"
#include "Buffer.h"
#include "TriIntersectorData.h"
#include "TriangleData.h"

namespace CudaTracerLib {

AABB ShapeSet::triData::box() const
{
	AABB b = AABB::Identity();
	for (unsigned int i = 0; i < 3; i++)
		b = b.Extend(p[i]);
	return b;
}

ShapeSet::ShapeSet(StreamReference<TriIntersectorData>* indices, BufferReference<TriangleData, TriangleData>* tri, unsigned int indexCount, const float4x4& mat, Stream<char>* buffer, Stream<TriIntersectorData>* triIntBuffer)
{
	count = indexCount;
	StreamReference<char> buffer2 = buffer->malloc_aligned<triData>(count * sizeof(triData));//buffer->malloc(count * sizeof(triData));
	StreamReference<char> buffer1 = buffer->malloc_aligned<float>((count + 1) * sizeof(float));//buffer->malloc((count + 1) * sizeof(float));
	areaDistribution = buffer1.AsVar<float>();
	triangles = buffer2.AsVar<triData>();
	for (unsigned int i = 0; i < count; i++)
	{
		triangles[i].iDat = indices[i].getIndex();
		triangles[i].tDat = tri[i].getIndex();
	}
	Recalculate(mat, buffer, triIntBuffer);
}

void ShapeSet::Recalculate(const float4x4& mat, Stream<char>* buffer, Stream<TriIntersectorData>* indices)
{
	buffer->translate(areaDistribution).Invalidate();
	buffer->translate(triangles).Invalidate();
	sumArea = 0;
	areaDistribution[0] = 0.0f;
	for (unsigned int i = 0; i < count; i++)
	{
		triangles[i].Recalculate(mat, *indices->operator()(triangles[i].iDat));
		sumArea += triangles[i].area;
		areaDistribution[i + 1] = areaDistribution[i] + triangles[i].area;
	}
	//pdf values have to be converted to normalized cdf
	for (unsigned int i = 0; i <= count; i++)
		areaDistribution[i] = areaDistribution[i] / sumArea;
}

}
