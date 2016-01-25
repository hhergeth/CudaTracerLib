#pragma once

#include <Math/float4x4.h>
#include <Math/AABB.h>

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
		unsigned int iDat;
		unsigned int tDat;

		AABB box() const;
		void Recalculate(const float4x4& mat, const TriIntersectorData& T);
	};
public:
	ShapeSet(){}
	ShapeSet(BufferReference<TriIntersectorData, TriIntersectorData>* indices, BufferReference<TriangleData, TriangleData>* triangles, unsigned int indexCount, const float4x4& mat, Stream<char>* buffer, Stream<TriIntersectorData>* triIntBuffer);
	CUDA_FUNC_IN float Area() const { return sumArea; }
	CUDA_DEVICE CUDA_HOST void SamplePosition(PositionSamplingRecord& pRec, const Vec2f& spatialSample, Vec2f* uv) const;
	CUDA_DEVICE CUDA_HOST bool getPosition(const Vec3f& pos, Vec2f* bary = 0, Vec2f* uv = 0) const;
	CUDA_DEVICE CUDA_HOST unsigned int sampleTriangle(Vec3f& p0, Vec3f& p1, Vec3f& p2, Vec2f& uv0, Vec2f& uv1, Vec2f& uv2, float& pdf, float sample) const;
	CUDA_DEVICE CUDA_HOST float PdfTriangle(const Vec3f& pos) const;
	CUDA_FUNC_IN float PdfPosition() const
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
	void Recalculate(const float4x4& mat, Stream<char>* buffer, Stream<TriIntersectorData>* indices);

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
