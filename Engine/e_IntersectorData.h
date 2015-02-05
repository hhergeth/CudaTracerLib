#pragma once

#define MAX_AREALIGHT_NUM 2

struct e_TriIntersectorData2
{
	unsigned int index;
public:
	CUDA_FUNC_IN void setIndex(unsigned int i)
	{
		index |= i << 1;
	}
	CUDA_FUNC_IN void setFlag(bool b)
	{
		index |= !!b;
	}
	CUDA_FUNC_IN unsigned int getIndex()
	{
		return index >> 1;
	}
	CUDA_FUNC_IN bool getFlag()
	{
		return index & 1;
	}
};

struct TraceResult;
struct e_TriIntersectorData
{
private:
	Vec4f a, b, c;
public:
	CUDA_DEVICE CUDA_HOST void setData(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2);

	CUDA_DEVICE CUDA_HOST void getData(Vec3f& v0, Vec3f& v1, Vec3f& v2) const;

	CUDA_DEVICE CUDA_HOST bool Intersect(const Ray& r, TraceResult* a_Result) const;
};