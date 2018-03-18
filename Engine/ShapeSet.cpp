#include <StdAfx.h>
#include "ShapeSet.h"
#include <Base/Buffer.h>
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

ShapeSet::ShapeSet(StreamReference<TriIntersectorData>* indices, BufferReference<TriangleData, TriangleData>* tri, unsigned int indexCount, const float4x4& mat, Stream<char>* buffer, Stream<TriIntersectorData>* triIntBuffer, Stream<TriangleData>* triDataBuffer)
{
	count = indexCount;
    StreamReference<char> buffer1 = buffer->malloc_aligned<float>((count + 1) * sizeof(float));//buffer->malloc((count + 1) * sizeof(float));
	StreamReference<char> buffer2 = buffer->malloc_aligned<triData>(count * sizeof(triData));//buffer->malloc(count * sizeof(triData));
    triData* triangles = (triData*)buffer2.operator char *();
	for (unsigned int i = 0; i < count; i++)
	{
		triangles[i].iDat = indices[i].getIndex();
		triangles[i].tDat = tri[i].getIndex();
	}

    m_areaDistributionIndex = buffer1.getIndex();
    m_areaDistributionLength = buffer1.getLength();
    m_trianglesIndex = buffer2.getIndex();
    m_trianglesLength = buffer2.getLength();

    Recalculate(mat, buffer, triIntBuffer, triDataBuffer);
}

void ShapeSet::Recalculate(const float4x4& mat, Stream<char>* buffer, Stream<TriIntersectorData>* indices, Stream<TriangleData>* triDataBuffer)
{
    StreamReference<char> buffer1 = buffer->operator()(m_areaDistributionIndex, m_areaDistributionLength);
    StreamReference<char> buffer2 = buffer->operator()(m_trianglesIndex, m_trianglesLength);

    buffer1.Invalidate();
    buffer2.Invalidate();

    triData* triangles = (triData*)buffer2.operator char *();
    float* areaDistribution = (float*)buffer1.operator char *();

	sumArea = 0;
	areaDistribution[0] = 0.0f;
	for (unsigned int i = 0; i < count; i++)
	{
		triangles[i].Recalculate(mat, *indices->operator()(triangles[i].iDat), *triDataBuffer->operator()(triangles[i].tDat));
		sumArea += triangles[i].area;
		areaDistribution[i + 1] = areaDistribution[i] + triangles[i].area;
	}
	//pdf values have to be converted to normalized cdf
	for (unsigned int i = 0; i <= count; i++)
		areaDistribution[i] = areaDistribution[i] / sumArea;
}

}
