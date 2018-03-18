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
		NormalizedT<Vec3f> n;
		float area;
		unsigned int iDat;
		unsigned int tDat;

        CTL_EXPORT CUDA_DEVICE CUDA_HOST AABB box() const;
		CTL_EXPORT void Recalculate(const float4x4& mat, const TriIntersectorData& T, const TriangleData& TData);
	};
public:
	ShapeSet(){}
	ShapeSet(BufferReference<TriIntersectorData, TriIntersectorData>* indices, BufferReference<TriangleData, TriangleData>* triangles, unsigned int indexCount, const float4x4& mat, Stream<char>* buffer, Stream<TriIntersectorData>* triIntBuffer, Stream<TriangleData>* triDataBuffer);
	CUDA_FUNC_IN float Area() const { return sumArea; }
	CTL_EXPORT CUDA_DEVICE CUDA_HOST void SamplePosition(PositionSamplingRecord& pRec, const Vec2f& spatialSample, Vec2f* uv) const;
	CTL_EXPORT CUDA_DEVICE CUDA_HOST bool getPosition(const Vec3f& pos, Vec2f* bary = 0, Vec2f* uv = 0) const;
	CTL_EXPORT CUDA_DEVICE CUDA_HOST unsigned int sampleTriangle(Vec3f& p0, Vec3f& p1, Vec3f& p2, Vec2f& uv0, Vec2f& uv1, Vec2f& uv2, float& pdf, float sample) const;
	CTL_EXPORT CUDA_DEVICE CUDA_HOST float PdfTriangle(const Vec3f& pos) const;
	CUDA_FUNC_IN float PdfPosition() const
	{
		return 1.0f / sumArea;
	}
    CTL_EXPORT CUDA_DEVICE CUDA_HOST AABB getBox() const;
	CTL_EXPORT void Recalculate(const float4x4& mat, Stream<char>* buffer, Stream<TriIntersectorData>* indices, Stream<TriangleData>* triDataBuffer);

	CUDA_FUNC_IN unsigned int numTriangles() const
	{
		return count;
	}
private:
    unsigned int m_areaDistributionIndex;
    unsigned int m_areaDistributionLength;
    unsigned int m_trianglesIndex;
    unsigned int m_trianglesLength;
	float sumArea;
	unsigned int count;
};

}
