#pragma once

#include <MathTypes.h>
#include "..\Math\half.h"
#include "e_Material.h"

#ifdef EXT_TRI
struct e_TriangleData
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
	e_TriangleData(){}
	e_TriangleData(const Vec3f* P, unsigned char matIndex, const Vec2f* T, const Vec3f* N, const Vec3f* Tan, const Vec3f* BiTan);
	CUDA_DEVICE CUDA_HOST void fillDG(const float4x4& localToWorld, const float4x4& worldToLocal, DifferentialGeometry& dg) const;
	CUDA_FUNC_IN unsigned int getMatIndex(const unsigned int off) const 
	{
		unsigned int v = (m_sDeviceData.NorMatExtra.y >> 16) & 0xff;
		return unsigned int(v) + off;
	}
	CUDA_DEVICE CUDA_HOST void setData(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2,
									   const Vec3f& n0, const Vec3f& n1, const Vec3f& n3);
	void setUvSetData(int setId, const Vec2f& a, const Vec2f& b, const Vec2f& c);
	CUDA_DEVICE CUDA_HOST void getNormals(Vec3f& n0, Vec3f& n1, Vec3f& n2);
};
#else
struct e_TriangleData
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
	e_TriangleData(){}
	e_TriangleData(Vec3f* P, unsigned char matIndex, Vec2f* T, Vec3f* N, Vec3f* Tan, Vec3f* BiTan)
	{
		Vec3f p = P[0] - P[2];
		Vec3f q = P[1] - P[2];
		Vec3f d = normalize(cross(p, q));
		m_sHostData.Normal = NormalizedFloat3ToUchar3(d);
		m_sHostData.MatIndex = matIndex;
	}

	CUDA_FUNC_IN Frame math::lerpFrame(const Vec2f& bCoords, const float4x4& localToWorld, Vec3f* ng = 0) const
	{
		Vec3f n = Uchar3ToNormalizedFloat3(m_sHostData.Normal);
		if(ng)
			*ng = n;
		return Frame(n);
	}
	CUDA_FUNC_IN unsigned int getMatIndex(const unsigned int off) const
	{
		unsigned int v = m_sDeviceData.Row0;
		return unsigned int(v >> 24) + off;
	}
	CUDA_FUNC_IN Vec2f math::lerpUV(const Vec2f& bCoords) const
	{
		return make_float2(0);
	}
	CUDA_FUNC_IN void getNormalDerivative(const Vec2f& bCoords, Vec3f& dndu, Vec3f& dndv) const
	{

	}
};
#endif