#pragma once

#include "..\Math\vector.h"
#include "e_Brdf.h"
#include "e_KernelMaterial.h"

#define EXT_TRI

#define TOFLOAT3(a,b,c) ((make_float3(a,b,c) / make_float3(127)) - make_float3(1))
#define TOUCHAR3(v) make_uchar3(unsigned char((v.x + 1) * 127.0f), unsigned char((v.y + 1) * 127.0f), unsigned char((v.z + 1) * 127.0f))

#ifdef EXT_TRI
struct e_TriangleData
{
private:
	union
	{
		struct
		{
			uchar3 Normals[3];
			uchar3 Tangents[3];
			uchar3 BiTangents[3];
			unsigned char MatIndex;
			ushort2 TexCoord[3];
		} m_sHostData;
		struct
		{
			uint4 Row0;
			uint3 Row1;
			uint3 Row2;
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
			m_sHostData.Normals[i] = TOUCHAR3(normalize(N[i]));
			m_sHostData.Tangents[i] = TOUCHAR3(normalize(Tan[i]));
			m_sHostData.BiTangents[i] = TOUCHAR3(normalize(BiTan[i]));
			m_sHostData.TexCoord[i] = *(ushort2*)&half2(T[i]);
			//NOR[i] = N[i];
		}
	}
	CUDA_FUNC_IN Onb lerpOnb(const float2& bCoords, const float4x4& localToWorld, float3* ng = 0) const 
	{
		uint4 q = m_sDeviceData.Row0;
		uint3 p = m_sDeviceData.Row1;
		float3 na = TOFLOAT3(q.x & 255, (q.x >> 8) & 255, (q.x >> 16) & 255), nb = TOFLOAT3(q.x >> 24, q.y & 255, (q.y >> 8) & 255), nc = TOFLOAT3((q.y >> 16) & 255, q.y >> 24, q.z & 255);
		//float3 na = NOR[0], nb = NOR[1], nc = NOR[2];
		float3 ta = TOFLOAT3((q.z >> 8) & 255, (q.z >> 16) & 255, q.z >> 24), tb = TOFLOAT3(q.w & 255, (q.w >> 8) & 255, (q.w >> 16) & 255), tc = TOFLOAT3(q.w >> 24, p.x & 255, (p.x >> 8) & 255);
		float3 ba = TOFLOAT3((p.x >> 16) & 255, p.x >> 24, p.y & 255), bb = TOFLOAT3((p.y >> 8) & 255, (p.y >> 16) & 255, p.y >> 24), bc = TOFLOAT3(p.z & 255, (p.z >> 8) & 255, (p.z >> 16) & 255);
		Onb sys;
		float w = 1.0f - bCoords.x - bCoords.y, u = bCoords.x, v = bCoords.y;
		sys.m_normal = (u * na + v * nb + w * nc);
		sys.m_tangent = (u * ta + v * tb + w * tc);
		sys.m_binormal = (u * ba + v * bb + w * bc);

		//TODO : Find out why this is necessary
		sys = Onb(sys.m_normal);

		sys = sys * localToWorld;
		if(ng)
			*ng = normalize(localToWorld.TransformNormal((na + nb + nc) / 3.0f));
		return sys;
	}
	CUDA_FUNC_IN unsigned int getMatIndex(const unsigned int off) const 
	{
		unsigned int v = m_sDeviceData.Row1.z;
		return unsigned int(v >> 24) + off;
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
		dat.i = m_sDeviceData.Row2.x;
		float2 a = dat.ToFloat2();
		dat.i = m_sDeviceData.Row2.y;
		float2 b = dat.ToFloat2();
		dat.i = m_sDeviceData.Row2.z;
		float2 c = dat.ToFloat2();
		float u = bCoords.y, v = 1.0f - u - bCoords.x;
		return a + u * (b - a) + v * (c - a);
	}
	CUDA_FUNC_IN e_KernelBSDF GetBSDF(const float2& baryCoords, const float4x4& localToWorld, const e_KernelMaterial* a_Mats, const unsigned int off) const 
	{
		float3 ng;
		Onb sys = lerpOnb(baryCoords, localToWorld, &ng);
		float2 uv = lerpUV(baryCoords);

		float3 nor;
		if(a_Mats[getMatIndex(off)].SampleNormalMap(uv, &nor))
			sys.m_normal = normalize(sys.localToworld(nor*2.0f-make_float3(1)));

		e_KernelBSDF bsdf(sys, ng);
		a_Mats[getMatIndex(off)].GetBSDF(uv, &bsdf);
		return bsdf;
	}
	CUDA_FUNC_IN float3 Le(const float2& baryCoords, const float3& n, const float3& w, const e_KernelMaterial* a_Mats, const unsigned int off) const 
	{
		return dot(w, n) > 0 ? a_Mats[getMatIndex(off)].Emission : make_float3(0);
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
	CUDA_FUNC_IN Onb lerpOnb(float2& bCoords)
	{
		unsigned int q = m_sDeviceData.Row0;
		Onb sys;
		sys.m_normal = TOFLOAT3(q & 255, (q >> 8) & 255, (q >> 16) & 255);
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

#undef TOFLOAT3
#undef TOUCHAR3