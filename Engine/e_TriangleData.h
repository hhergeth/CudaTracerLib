#pragma once

#include <MathTypes.h>
#include "..\Math\half.h"
#include "e_Material.h"
#include "e_TraceResult.h"

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
			unsigned short Normals[3];//Row0.x,y
			unsigned short dpdu, dpdv, faceN;//Row0.y,z
			unsigned char MatIndex;//Row0.w
			unsigned char ExtraData[3];//Row0.w
			UV_Set UV_Sets[NUM_UV_SETS];
		} m_sHostData;
		struct
		{
			uint4 Row0;
			uint3 RowX[NUM_UV_SETS];
		} m_sDeviceData;
	};
	float3 dpdu, dpdv;
public:
	e_TriangleData(){}
	e_TriangleData(const float3* P, unsigned char matIndex, const float2* T, const float3* N, const float3* Tan, const float3* BiTan);
	CUDA_DEVICE CUDA_HOST void fillDG(const float4x4& localToWorld, const float4x4& worldToLocal, DifferentialGeometry& dg) const;
	CUDA_FUNC_IN unsigned int getMatIndex(const unsigned int off) const 
	{
		unsigned int v = m_sDeviceData.Row0.w & 0xff;
		return unsigned int(v) + off;
	}
	CUDA_DEVICE CUDA_HOST float2 lerpUV(int setId, const float2& bCoords) const;
	CUDA_DEVICE CUDA_HOST void getNormalDerivative(const float2& bCoords, float3& dndu, float3& dndv) const;
	CUDA_DEVICE CUDA_HOST void setData(const float3& v0, const float3& v1, const float3& v2,
									   const float3& n0, const float3& n1, const float3& n3);
	CUDA_FUNC_IN unsigned char lerpExtraData(const float2& bCoords) const
	{
		unsigned int V = m_sDeviceData.Row0.w;
		unsigned char a = (V >> 8) & 0xff, b = (V >> 16) & 0xff, c = V >> 24;
		float u = bCoords.y, v = 1.0f - u - bCoords.x;
		return unsigned char(a + u * (b - a) + v * (c - a));
	}
	void setUvSetData(int setId, const float2& a, const float2& b, const float2& c)
	{
		half2 a1 = half2(a), b1 = half2(b), c1 = half2(c);
		m_sHostData.UV_Sets[setId].TexCoord[0] = *(ushort2*)&a1;
		m_sHostData.UV_Sets[setId].TexCoord[1] = *(ushort2*)&b1;
		m_sHostData.UV_Sets[setId].TexCoord[2] = *(ushort2*)&c1;
	}
	//CUDA_DEVICE CUDA_HOST void getCurvature(const float2& bCoords, float& H, float& K) const;
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
	e_TriangleData(float3* P, unsigned char matIndex, float2* T, float3* N, float3* Tan, float3* BiTan)
	{
		float3 p = P[0] - P[2];
		float3 q = P[1] - P[2];
		float3 d = normalize(cross(p, q));
		m_sHostData.Normal = NormalizedFloat3ToUchar3(d);
		m_sHostData.MatIndex = matIndex;
	}

	CUDA_FUNC_IN Frame lerpFrame(const float2& bCoords, const float4x4& localToWorld, float3* ng = 0) const
	{
		float3 n = Uchar3ToNormalizedFloat3(m_sHostData.Normal);
		if(ng)
			*ng = n;
		return Frame(n);
	}
	CUDA_FUNC_IN unsigned int getMatIndex(const unsigned int off) const
	{
		unsigned int v = m_sDeviceData.Row0;
		return unsigned int(v >> 24) + off;
	}
	CUDA_FUNC_IN float2 lerpUV(const float2& bCoords) const
	{
		return make_float2(0);
	}
	CUDA_FUNC_IN void getNormalDerivative(const float2& bCoords, float3& dndu, float3& dndv) const
	{

	}
};
#endif