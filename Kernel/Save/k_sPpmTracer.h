#pragma once

#include "k_Tracer.h"
#include "..\Base\CudaRandom.h"
#include "..\Engine\e_PointBVH.h"
#include "..\Base\Timer.h"
#include <time.h>

#define ALPHA 0.7f

struct k_sPpmEntry
{
	float3 Weight;
	float3 Pos;
	float3 Tau;
	float3 Nor;
	float3 Dir;
	unsigned int next;
	const e_KernelBXDF* Bsdf;
	CUDA_FUNC_IN void setData(float3 w, float3 p, float3 n, float3 d, const e_KernelBXDF* b, unsigned int ne)
	{
		next = ne;
		Dir = d;
		Bsdf = b;
		Weight = w;
		Pos = p;
		Nor = n;
	}
	k_sPpmEntry(){}
};

struct k_sPpmPixel
{
	//float3 m_vPixelColor;
};

#define INVALID_HASH -1

#define k_HashGrid k_HashGrid_Irreg

struct k_HashGrid_Irreg
{
	float HashScale;
	float HashNum;
	float3 m_vMin;
	AABB m_sBox;

	CUDA_FUNC_IN k_HashGrid_Irreg(AABB box, float a_InitialRadius, unsigned int a_NumEntries)
	{
		m_sBox = box.Enlarge();
		HashScale = 1.0f / (a_InitialRadius * 1.5f);
		HashNum = a_NumEntries;
		m_vMin = m_sBox.minV;
	}

	CUDA_FUNC_IN unsigned int Hash(uint3& p)
	{
		return hashp(make_float3(p.x,p.y,p.z));
	}

	CUDA_FUNC_IN uint3 Transform(float3& p)
	{
		return make_uint3(fabsf(p - m_vMin) * HashScale);
	}

	CUDA_FUNC_IN bool IsValidHash(float3& p)
	{
		return m_sBox.Contains(p);
	} 
private:
	CUDA_FUNC_IN unsigned int hashp(const float3 idx)
	{
		// use the same procedure as GPURnd
		float4 n = make_float4(idx, idx.x + idx.y - idx.z) * 4194304.0 / HashScale;

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
	int m_fGridSize;
	float3 m_vMin;
	float3 m_vInvSize;

	CUDA_FUNC_IN k_HashGrid_Reg(AABB box, float a_InitialRadius, unsigned int a_NumEntries)
	{
		float3 q = (box.maxV - box.minV) / 2.0f, m = (box.maxV + box.minV) / 2.0f;
		float e = 0.015f, e2 = 1.0f + e;
		box.maxV = m + q * e2;
		box.minV = m - q * e2;
		m_fGridSize = (int)floor(sqrtf(sqrtf(a_NumEntries)));
		m_vMin = box.minV;
		m_vInvSize = make_float3(1.0f) / box.Size() * m_fGridSize;
	}

	CUDA_FUNC_IN unsigned int Hash(uint3& p)
	{
		return (unsigned int)(p.z * m_fGridSize * m_fGridSize + p.y * m_fGridSize + p.x);
	}

	CUDA_FUNC_IN  uint3 Transform(float3& p)
	{
		return make_uint3((p - m_vMin) * m_vInvSize);
	}

	CUDA_FUNC_IN bool IsValidHash(float3& p)
	{
		uint3 q = Transform(p);
		return q.x >= 0 && q.x <= m_fGridSize && q.y >= 0 && q.y <= m_fGridSize && q.z >= 0 && q.z <= m_fGridSize;
	}
};

class k_sPpmTracer : public k_RandTracerBase
{
private:
	k_sPpmEntry* m_pDeviceEntries;
	k_sPpmPixel* m_pDevicePixels;
	unsigned int* m_pDeviceHashGrid;
	k_HashGrid m_sHash;

	unsigned int m_uCurrentRunIndex;

	float m_fCurrentRadius;
	unsigned int m_uCurrentEyePassIndex;
	unsigned long long m_uPhotonsEmitted;

	float m_fInitialRadiusScale;
	const unsigned int m_uGridLength;
	const unsigned int m_uNumRunsPerEyePass;
public:
	k_sPpmTracer()
		: k_RandTracerBase(), m_sHash(AABB::Identity(), 1, 1), m_uGridLength(200*200*200), m_uNumRunsPerEyePass(20)
	{
		m_pDevicePixels = 0;
		m_pDeviceEntries = 0;
		m_fInitialRadiusScale = 1;
		cudaMalloc(&m_pDeviceHashGrid, sizeof(unsigned int) * m_uGridLength);
	}
	virtual ~k_sPpmTracer()
	{
		cudaFree(m_pDeviceEntries);
		cudaFree(m_pDevicePixels);
		cudaFree(m_pDeviceHashGrid);
	}
	virtual void Resize(unsigned int _w, unsigned int _h)
	{
		k_TracerBase::Resize(_w, _h);
		if(m_pDeviceEntries)
			cudaFree(m_pDeviceEntries);
		if(m_pDevicePixels)
			cudaFree(m_pDevicePixels);
		cudaMalloc(&m_pDeviceEntries, w * h * sizeof(k_sPpmEntry));
		cudaMalloc(&m_pDevicePixels, w * h * sizeof(k_sPpmPixel));
	}
	virtual void Debug(int2 pixel);
	virtual void PrintStatus(std::vector<FW::String>& a_Buf)
	{
		double pC = floor((double)m_uPhotonsEmitted / 1000000.0);
		a_Buf.push_back(FW::sprintf("Photons emitted : %d[Mil]", (int)pC));
		double pCs = getValuePerSecond(m_uPhotonsEmitted, 1000000.0);
		a_Buf.push_back(FW::sprintf("Photons/Sec : %f", (float)pCs));
	}
	float* getRadiusScalePointer()
	{
		return &m_fInitialRadiusScale;
	}
	virtual void CreateSliders(SliderCreateCallback a_Callback)
	{
		a_Callback(0.1f, 10.0f, true, &m_fInitialRadiusScale, "Initial radius = %g units");
	}
protected:
	virtual void DoRender(RGBCOL* a_Buf);
	virtual void StartNewTrace();
};