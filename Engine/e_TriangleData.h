#pragma once

#include <MathTypes.h>
#include "..\Math\half.h"
#include "e_Material.h"

#define EXT_TRI

struct e_TriangleData;
class e_Node;
struct e_KernelBSDF;
struct e_KernelMaterial;
struct e_KernelBSSRDF;
struct TraceResult
{
	float m_fDist;
	float2 m_fUV;
	const e_TriangleData* m_pTri;
	const e_Node* m_pNode;
	CUDA_FUNC_IN bool hasHit() const
	{
		return m_pTri != 0;
	}
	CUDA_FUNC_IN void Init()
	{
		m_fDist = FLT_MAX;
		m_fUV = make_float2(0,0);
		m_pNode = 0;
		m_pTri = 0;
	}
	CUDA_FUNC_IN operator bool() const
	{
		return hasHit();
	}

	CUDA_FUNC_IN Frame lerpFrame() const;
	CUDA_FUNC_IN unsigned int getMatIndex() const;
	CUDA_FUNC_IN float2 lerpUV() const;
	CUDA_FUNC_IN float3 Le(const float3& p, const float3& n, const float3& w) const;
	CUDA_FUNC_IN unsigned int LightIndex() const;
	CUDA_FUNC_IN const e_KernelMaterial& getMat() const;
	CUDA_FUNC_IN void getBsdfSample(const Ray& r, CudaRNG _rng, BSDFSamplingRecord* bRec) const;
};

#ifdef EXT_TRI
struct e_TriangleData
{
public:
	union
	{
		struct
		{
			uchar2 Normals[3];//Row0.x,y
			uchar2 Tangents[3];//Row0.y,z
			unsigned int MatIndex;//Row0.w
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
	e_TriangleData(float3* P, unsigned char matIndex, float2* T, float3* N, float3* Tan, float3* BiTan)
	{
		m_sHostData.MatIndex = matIndex;
		for(int i = 0; i < 3; i++)
		{
			m_sHostData.Normals[i] = NormalizedFloat3ToUchar2(normalize(N[i]));
			m_sHostData.Tangents[i] = NormalizedFloat3ToUchar2(normalize(Tan[i]));
			m_sHostData.TexCoord[i] = *(ushort2*)&half2(T[i]);
			//NOR[i] = N[i];
		}
	}
	CUDA_FUNC_IN Frame lerpFrame(const float2& bCoords, const float4x4& localToWorld, float3* ng = 0) const 
	{
		//float3 na = NOR[0], nb = NOR[1], nc = NOR[2];
		uint4 q = m_sDeviceData.Row0;
		float3 na = Uchar2ToNormalizedFloat3(q.x), nb = Uchar2ToNormalizedFloat3(q.x >> 16), nc = Uchar2ToNormalizedFloat3(q.y);
		float3 ta = Uchar2ToNormalizedFloat3(q.y >> 16), tb = Uchar2ToNormalizedFloat3(q.z), tc = Uchar2ToNormalizedFloat3(q.z >> 16);
		Frame sys;
		float w = 1.0f - bCoords.x - bCoords.y, u = bCoords.x, v = bCoords.y;
		sys.n = (u * na + v * nb + w * nc);
		sys.t = (u * ta + v * tb + w * tc);

		sys = sys * localToWorld;
		sys.s = normalize(cross(sys.t, sys.n));
		sys.t = normalize(cross(sys.s, sys.n));
		if(ng)
			*ng = normalize(localToWorld.TransformNormal((na + nb + nc) / 3.0f));
		return sys;
	}
	CUDA_FUNC_IN unsigned int getMatIndex(const unsigned int off) const 
	{
		unsigned int v = m_sDeviceData.Row0.w;
		return unsigned int(v ) + off;
	}
	CUDA_FUNC_IN float2 lerpUV(const float2& bCoords) const 
	{
		union
		{
			ushort2 h;
			unsigned int i;
			CUDA_FUNC_IN float2 ToFloat2()
			{
				return make_float2(half(h.x).ToFloat(), half(h.y).ToFloat());
			}
		} dat;
		dat.i = m_sDeviceData.Row1.x;
		float2 a = dat.ToFloat2();
		dat.i = m_sDeviceData.Row1.y;
		float2 b = dat.ToFloat2();
		dat.i = m_sDeviceData.Row1.z;
		float2 c = dat.ToFloat2();
		float u = bCoords.y, v = 1.0f - u - bCoords.x;
		return a + u * (b - a) + v * (c - a);
	}
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
		m_sHostData.Normal = TOUCHAR3(d);
		m_sHostData.MatIndex = matIndex;
	}
	CUDA_FUNC_IN Frame lerpFrame(float2& bCoords)
	{
		unsigned int q = m_sDeviceData.Row0;
		Frame sys;
		sys.n = TOFLOAT3(q & 255, (q >> 8) & 255, (q >> 16) & 255);
		return sys;
	}
	CUDA_FUNC_IN unsigned int getMatIndex()
	{
		unsigned int v = m_sDeviceData.Row0;
		return v >> 24;
	}
	CUDA_FUNC_IN float2 lerpUV(float2& bCoords)
	{
		return make_float2(0);
	}
};
#endif