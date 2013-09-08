#pragma once

#include "..\Math\AABB.h"

struct k_HashGrid_Irreg
{
	float HashScale;
	float HashNum;
	float3 m_vMin;
	AABB m_sBox;

	CUDA_FUNC_IN k_HashGrid_Irreg(){}

	CUDA_FUNC_IN k_HashGrid_Irreg(const AABB& box, float a_InitialRadius, unsigned int a_NumEntries)
	{
		m_sBox = box.Enlarge();
		HashScale = 1.0f / (a_InitialRadius * 1.5f);
		HashNum = a_NumEntries;
		m_vMin = m_sBox.minV;
	}

	CUDA_FUNC_IN unsigned int Hash(const uint3& p) const
	{
		return hashp(make_float3(p.x,p.y,p.z));
	}

	CUDA_FUNC_IN uint3 Transform(const float3& p) const
	{
		return make_uint3(fabsf(p - m_vMin) * HashScale);
	}

	CUDA_FUNC_IN bool IsValidHash(const float3& p) const
	{
		return m_sBox.Contains(p);
	}

	CUDA_FUNC_IN unsigned int EncodePos(const float3& p, const uint3& i) const
	{
		float3 low = make_float3(i.x, i.y, i.z) / HashScale + m_vMin;
		float3 m = (p - low) * HashScale * 255.0f;
		return (unsigned int(m.x) << 16) | (unsigned int(m.y) << 8) | (unsigned int(m.z));
	}
	
	CUDA_FUNC_IN float3 DecodePos(unsigned int p, const uint3& i) const
	{
		float3 low = make_float3(i.x, i.y, i.z) / HashScale + m_vMin;
		return (make_float3(p >> 16, (p >> 8) & 0xff, p & 0xff) / 255.0f) / HashScale + low;
	}

	CUDA_FUNC_IN AABB getAABB() const
	{
		return m_sBox;
	}
private:
	CUDA_FUNC_IN unsigned int hashp(const float3 idx) const
	{
		// use the same procedure as GPURnd
		float4 n = make_float4(idx, idx.x + idx.y - idx.z) * 4194304.0;// / HashScale

		const float4 q = make_float4(   1225.0,    1585.0,    2457.0,    2098.0);
		const float4 r = make_float4(   1112.0,     367.0,      92.0,     265.0);
		const float4 a = make_float4(   3423.0,    2646.0,    1707.0,    1999.0);
		const float4 m = make_float4(4194287.0, 4194277.0, 4194191.0, 4194167.0);

		float4 beta = floor(n / q);
		float4 p = a * (n - beta * q) - beta * r;
		beta = (signf(-p) + make_float4(1.0)) * make_float4(0.5) * m;
		n = (p + beta);

		return (unsigned int)floor( frac(dot(n / m, make_float4(1.0, -1.0, 1.0, -1.0))) * HashNum );
	}
};

struct k_HashGrid_Reg
{
	unsigned int m_fGridSize;
	float3 m_vMin;
	float3 m_vInvSize;
	float3 m_vCellSize;
	AABB m_sBox;

	CUDA_FUNC_IN k_HashGrid_Reg(){}

	CUDA_FUNC_IN k_HashGrid_Reg(const AABB& box, float a_InitialRadius, unsigned int a_NumEntries)
	{
		float3 q = (box.maxV - box.minV) / 2.0f, m = (box.maxV + box.minV) / 2.0f;
		float e = 0.015f, e2 = 1.0f + e;
		m_sBox.maxV = m + q * e2;
		m_sBox.minV = m - q * e2;
		m_fGridSize = (int)floor(pow(a_NumEntries, 1.0/3.0));
		m_vMin = m_sBox.minV;
		m_vInvSize = make_float3(1.0f) / m_sBox.Size() * m_fGridSize;
		m_vCellSize = m_sBox.Size() / m_fGridSize;
	}

	CUDA_FUNC_IN unsigned int Hash(const uint3& p) const
	{
		return (unsigned int)(p.z * m_fGridSize * m_fGridSize + p.y * m_fGridSize + p.x);
	}

	CUDA_FUNC_IN  uint3 Transform(const float3& p) const
	{
		return clamp(make_uint3((p - m_vMin) * m_vInvSize), 0, m_fGridSize);
	}

	CUDA_FUNC_IN bool IsValidHash(const float3& p) const
	{
		uint3 q = Transform(p);
		return q.x <= m_fGridSize && q.y <= m_fGridSize && q.z <= m_fGridSize;
	}

	CUDA_FUNC_IN unsigned int EncodePos(const float3& p, const uint3& i) const
	{
		float3 low = make_float3(i.x, i.y, i.z) / m_vInvSize + m_vMin;
		float3 m = saturate((p - low) / m_vCellSize) * 255.0f;
		return (unsigned int(m.x) << 16) | (unsigned int(m.y) << 8) | (unsigned int(m.z));
	}
	
	CUDA_FUNC_IN float3 DecodePos(unsigned int p, const uint3& i) const
	{
		const unsigned int q = 0x00ff0000, q2 = 0x0000ff00, q3 = 0x000000ff;
		float3 low = make_float3(i.x, i.y, i.z) / m_vInvSize + m_vMin;
		float3 m = (make_float3((p & q) >> 16, (p & q2) >> 8, (p & q3)) / 255.0f) * m_vCellSize + low;
		return m;
	}

	CUDA_FUNC_IN AABB getAABB() const
	{
		return m_sBox;
	}
};