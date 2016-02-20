#pragma once

#include <Math/float4x4.h>
#include "Material.h"

namespace CudaTracerLib {

#ifdef EXT_TRI
struct TriangleData
{
public:

	struct UV_Set
	{
		ushort2 TexCoord[3];
	};

	union
	{
		struct
		{
			unsigned short Normals[3];
			unsigned char MatIndex;
			unsigned char ExtraData;
			ushort3 dpdu, dpdv;
			UV_Set UV_Sets[NUM_UV_SETS];
		} m_sHostData;
		struct
		{
			uint2 NorMatExtra;
			uint3 DpduDpdv;
			uint3 UVSets[NUM_UV_SETS];
		} m_sDeviceData;
	};
public:
	TriangleData(){}
	CTL_EXPORT TriangleData(const Vec3f* P, unsigned char matIndex, const Vec2f* T, const NormalizedT<Vec3f>* N, const Vec3f* Tan, const Vec3f* BiTan);
	CTL_EXPORT CUDA_DEVICE CUDA_HOST void fillDG(const float4x4& localToWorld, DifferentialGeometry& dg) const;
	CUDA_FUNC_IN unsigned int getMatIndex(const unsigned int off) const
	{
		unsigned int v = (m_sDeviceData.NorMatExtra.y >> 16) & 0xff;
		return v + off;
	}
	CTL_EXPORT CUDA_DEVICE CUDA_HOST void setData(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2,
									   const NormalizedT<Vec3f>& n0, const NormalizedT<Vec3f>& n1, const NormalizedT<Vec3f>& n3);
	CTL_EXPORT void setUvSetData(int setId, const Vec2f& a, const Vec2f& b, const Vec2f& c);
	CTL_EXPORT CUDA_DEVICE CUDA_HOST void getUVSetData(int setId, Vec2f& a, Vec2f& b, Vec2f& c) const;
	CTL_EXPORT CUDA_DEVICE CUDA_HOST void getNormals(NormalizedT<Vec3f>& n0, NormalizedT<Vec3f>& n1, NormalizedT<Vec3f>& n2) const;
};
#else
struct TriangleData
{
	union
	{
		struct
		{
			uchar3 Normal;
			unsigned char MatIndex;
		} m_sHostData;
		struct
		{
			unsigned int Row0;
		} m_sDeviceData;
	};
	TriangleData(){}
	CTL_EXPORT TriangleData(const Vec3f* P, unsigned char matIndex, const Vec2f* T, const NormalizedT<Vec3f>* N, const Vec3f* Tan, const Vec3f* BiTan);
	CTL_EXPORT CUDA_DEVICE CUDA_HOST void fillDG(const float4x4& localToWorld, DifferentialGeometry& dg) const;
	CUDA_FUNC_IN unsigned int getMatIndex(const unsigned int off) const
	{
		unsigned int v = m_sDeviceData.Row0;
		return unsigned int(v >> 24) + off;
	}
	CTL_EXPORT CUDA_DEVICE CUDA_HOST void setData(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2,
									   const NormalizedT<Vec3f>& n0, const NormalizedT<Vec3f>& n1, const NormalizedT<Vec3f>& n3);
	void setUvSetData(int setId, const Vec2f& a, const Vec2f& b, const Vec2f& c)
	{

	}
	void getUVSetData(int setId, Vec2f& a, Vec2f& b, Vec2f& c) const
	{
		a = b = c = Vec2f(0.0f);
	}
	CTL_EXPORT CUDA_DEVICE CUDA_HOST void getNormals(NormalizedT<Vec3f>& n0, NormalizedT<Vec3f>& n1, NormalizedT<Vec3f>& n2) const;
};
#endif

}
