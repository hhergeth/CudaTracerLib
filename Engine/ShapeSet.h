#pragma once

#include <MathTypes.h>

namespace CudaTracerLib {

struct TriIntersectorData;
struct PositionSamplingRecord;
template<typename H, typename D> class BufferReference;
template<typename T> class Stream;

struct CUDA_ALIGN(16) ShapeSet
{
	CUDA_ALIGN(16) struct triData
	{
		CUDA_ALIGN(16) Vec3f p[3];
		float area;
		e_Variable<TriIntersectorData> iDat;

		AABB box() const;
		void Recalculate(const float4x4& mat);
	};
public:
	ShapeSet(){}
	ShapeSet(BufferReference<TriIntersectorData, TriIntersectorData>* indices, unsigned int indexCount, float4x4& mat, Stream<char>* buffer);
	CUDA_FUNC_IN float Area() const { return sumArea; }
	CUDA_DEVICE CUDA_HOST void SamplePosition(PositionSamplingRecord& pRec, const Vec2f& spatialSample) const;
	CUDA_FUNC_IN float Pdf() const
	{
		return 1.0f / sumArea;
	}
	CUDA_FUNC_IN AABB getBox() const
	{
		AABB b = AABB::Identity();
		for (unsigned int i = 0; i < count; i++)
			b = b.Extend(triangles[i].box());
		return b;
	}
	void Recalculate(const float4x4& mat, Stream<char>* buffer);

	CUDA_FUNC_IN unsigned int numTriangles() const
	{
		return count;
	}
	CUDA_FUNC_IN const triData& getTriangle(unsigned int index) const
	{
		return triangles[index];
	}
private:
	e_Variable<float> areaDistribution;
	e_Variable<triData> triangles;
	float sumArea;
	unsigned int count;
};

}