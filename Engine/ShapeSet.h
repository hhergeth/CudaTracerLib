#pragma once

#include <MathTypes.h>

namespace CudaTracerLib {

struct TriIntersectorData;
struct TriangleData;
struct PositionSamplingRecord;
template<typename H, typename D> class BufferReference;
template<typename T> class Stream;

struct ShapeSet
{
	struct CUDA_ALIGN(16) triData
	{
		Vec3f p[3];
		float area;
		e_Variable<TriIntersectorData> iDat;
		e_Variable<TriangleData> tDat;

		AABB box() const;
		void Recalculate(const float4x4& mat);
	};
public:
	ShapeSet(){}
	ShapeSet(BufferReference<TriIntersectorData, TriIntersectorData>* indices, BufferReference<TriangleData, TriangleData>* triangles, unsigned int indexCount, const float4x4& mat, Stream<char>* buffer);
	CUDA_FUNC_IN float Area() const { return sumArea; }
	CUDA_DEVICE CUDA_HOST void SamplePosition(PositionSamplingRecord& pRec, const Vec2f& spatialSample, Vec2f* uv) const;
	CUDA_DEVICE CUDA_HOST bool getPosition(const Vec3f& pos, Vec2f* bary = 0, Vec2f* uv = 0) const;
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
