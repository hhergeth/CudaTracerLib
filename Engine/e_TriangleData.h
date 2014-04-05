#pragma once

#include <MathTypes.h>
#include "..\Math\half.h"
#include "e_Material.h"
#include "e_TraceResult.h"

#define EXT_TRI

#ifdef EXT_TRI
struct e_TriangleData
{
public:
	union
	{
		struct
		{
			unsigned short Normals[3];//Row0.x,y
			unsigned short Tangents[3];//Row0.y,z
			unsigned char MatIndex;//Row0.w
			unsigned char ExtraData[3];//Row0.w
			ushort2 TexCoord[3];//Row1.x,y,z
		} m_sHostData;
		struct
		{
			uint4 Row0;
			uint3 Row1;
		} m_sDeviceData;
	};
	//float3 NOR[3];
public:
	e_TriangleData(){}
	e_TriangleData(float3* P, unsigned char matIndex, float2* T, float3* N, float3* Tan, float3* BiTan);
	CUDA_DEVICE CUDA_HOST void lerpFrame(const float2& bCoords, const float4x4& localToWorld, Frame& sys, float3* ng = 0) const;
	CUDA_FUNC_IN unsigned int getMatIndex(const unsigned int off) const 
	{
		unsigned int v = m_sDeviceData.Row0.w & 0xff;
		return unsigned int(v) + off;
	}
	CUDA_DEVICE CUDA_HOST float2 lerpUV(const float2& bCoords) const;
	CUDA_DEVICE CUDA_HOST void getNormalDerivative(const float2& bCoords, float3& dndu, float3& dndv) const;
	CUDA_DEVICE CUDA_HOST void setData(const float3& na, const float3& nb, const float3& nc,
									   const float3& ta, const float3& tb, const float3& tc);
	CUDA_FUNC_IN unsigned char lerpExtraData(const float2& bCoords) const
	{
		unsigned int V = m_sDeviceData.Row0.w;
		unsigned char a = (V >> 8) & 0xff, b = (V >> 16) & 0xff, c = V >> 24;
		float u = bCoords.y, v = 1.0f - u - bCoords.x;
		return unsigned char(a + u * (b - a) + v * (c - a));
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