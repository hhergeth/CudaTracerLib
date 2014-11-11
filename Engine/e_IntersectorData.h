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
	float4 a, b, c;
public:
	CUDA_DEVICE CUDA_HOST void setData(const float3& v0, const float3& v1, const float3& v2);

	CUDA_DEVICE CUDA_HOST void getData(float3& v0, float3& v1, float3& v2) const;

	CUDA_DEVICE CUDA_HOST bool Intersect(const Ray& r, TraceResult* a_Result) const;
};