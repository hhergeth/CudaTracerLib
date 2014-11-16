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
	e_TriangleData(const float3* P, unsigned char matIndex, const float2* T, const float3* N, const float3* Tan, const float3* BiTan);
	CUDA_DEVICE CUDA_HOST void fillDG(const float4x4& localToWorld, const float4x4& worldToLocal, DifferentialGeometry& dg) const;
	CUDA_FUNC_IN unsigned int getMatIndex(const unsigned int off) const 
	{
		unsigned int v = (m_sDeviceData.NorMatExtra.y >> 16) & 0xff;
		return unsigned int(v) + off;
	}
	CUDA_DEVICE CUDA_HOST void setData(const float3& v0, const float3& v1, const float3& v2,
									   const float3& n0, const float3& n1, const float3& n3);
	void setUvSetData(int setId, const float2& a, const float2& b, const float2& c);
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