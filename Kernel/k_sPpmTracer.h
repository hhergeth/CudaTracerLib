#pragma once

#include "k_Tracer.h"
#include "..\Base\CudaRandom.h"
#include "..\Engine\e_PointBVH.h"
#include "..\Base\Timer.h"
#include <time.h>

#define ALPHA (2.0f / 3.0f)

struct k_pPpmPhoton
{
	float3 Pos;
	RGBE L;
	uchar2 Wi;
	uchar2 Nor;
	unsigned int next;
	CUDA_FUNC_IN k_pPpmPhoton(){}
	CUDA_FUNC_IN k_pPpmPhoton(float3 p, float3 l, float3 wi, float3 n, unsigned int ne)
	{
		Pos = p;
		Nor = NormalizedFloat3ToUchar2(n);
		L = Float3ToRGBE(l);
		Wi = NormalizedFloat3ToUchar2(wi);
		next = ne;
	}
	CUDA_FUNC_IN float3 getNormal()
	{
		return Uchar2ToNormalizedFloat3(Nor);
	}
	CUDA_FUNC_IN float3 getWi()
	{
		return Uchar2ToNormalizedFloat3(Wi);
	}
	CUDA_FUNC_IN float3 getL()
	{
		return RGBEToFloat3(L);
	}
	CUDA_FUNC_IN uint4 toUint4()
	{/*
		uint4 r;
		r.x = Pos;
		r.y = (L.x << 24) | (L.y << 16) | (L.z << 8) | L.w;
		r.z = (L.x << 24) | (L.y << 16) | (L.z << 8) | L.w;
		r.w = next;
		return r;*/
		return *(uint4*)this;
	}/*
	CUDA_FUNC_IN k_pPpmPhoton(const uint4& v)
	{
		Pos = v.x;
		L = *(RGBE*)&v.y;
		Wi = make_uchar2(v.z & 0xff, (v.z >> 8) & 0xff);
		Nor = make_uchar2((v.z >> 16) & 0xff, v.z >> 24);
		next = v.w;
	}*/
};

struct k_sPpmPixel
{
	float3 m_vPixelColor;
};

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
		return q.x >= 0 && q.x <= m_fGridSize && q.y >= 0 && q.y <= m_fGridSize && q.z >= 0 && q.z <= m_fGridSize;
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

template<typename HASH> struct k_PhotonMap
{
	k_pPpmPhoton* m_pDevicePhotons;
	unsigned int* m_pDeviceHashGrid;
	HASH m_sHash;
	unsigned int m_uMaxPhotonCount;
	unsigned int m_uGridLength;

	k_PhotonMap()
	{

	}

	k_PhotonMap(unsigned int photonN, unsigned int hashN, k_pPpmPhoton* P)
	{
		m_uGridLength = hashN;
		m_pDevicePhotons = P;
		m_pDeviceHashGrid = 0;
		cudaMalloc(&m_pDeviceHashGrid, sizeof(unsigned int) * m_uGridLength);
		cudaMemset(m_pDeviceHashGrid, -1, sizeof(unsigned int) * m_uGridLength);
		m_uMaxPhotonCount = photonN;
	}

	void Free()
	{
		cudaFree(m_pDeviceHashGrid);
	}

	void StartNewPass()
	{
		cudaMemset(m_pDeviceHashGrid, -1, sizeof(unsigned int) * m_uGridLength);
	}

	void Resize(unsigned int N, k_pPpmPhoton* P)
	{
		m_pDevicePhotons = P;
		m_uMaxPhotonCount = N;
	}

	void StartNewRendering(const AABB& box, float a_InitRadius);

#ifdef __CUDACC__
	CUDA_ONLY_FUNC bool StorePhoton(const float3& p, const float3& l, const float3& wi, const float3& n, unsigned int* a_PhotonCounter) const
	{
		if(!m_sHash.IsValidHash(p))
			return false;
		uint3 i0 = m_sHash.Transform(p);
		unsigned int i = m_sHash.Hash(i0);
		unsigned int j = atomicInc(a_PhotonCounter, -1);
		if(j < m_uMaxPhotonCount)
		{
			unsigned int k = atomicExch(m_pDeviceHashGrid + i, j);
			m_pDevicePhotons[j] = k_pPpmPhoton(p, l, wi, n, k);//m_sHash.EncodePos(p, i0)
			return true;
		}
		return false;
	}

	CUDA_ONLY_FUNC float3 L_Surface(float a_r, float a_NumPhotonEmitted, CudaRNG& rng, const e_KernelBSDF* bsdf, const float3& n, const float3& p, const float3& wo) const;

	CUDA_ONLY_FUNC float3 L_Volume(float a_r, float a_NumPhotonEmitted, CudaRNG& rng, const Ray& r, float tmin, float tmax, const float3& Li) const;
#endif
};

struct k_PhotonMapCollection
{
	k_PhotonMap<k_HashGrid_Reg> m_sVolumeMap;
	k_PhotonMap<k_HashGrid_Irreg> m_sSurfaceMap;
	k_pPpmPhoton* m_pPhotons;
	unsigned int m_uPhotonBufferLength;
	unsigned int m_uPhotonNumStored;
	unsigned int m_uPhotonNumEmitted;
	unsigned int m_uRealBufferSize;

	CUDA_FUNC_IN k_PhotonMapCollection()
	{

	}

	k_PhotonMapCollection(unsigned int a_BufferLength, unsigned int a_HashNum);

	void Free();

	void Resize(unsigned int a_BufferLength);

	void StartNewPass();

	bool PassFinished();
#ifdef __CUDACC__

	void StartNewRendering(const AABB& sbox, const AABB& vbox, float a_R);

	template<bool SURFACE> CUDA_ONLY_FUNC bool StorePhoton(const float3& p, const float3& l, const float3& wi, const float3& n)
	{
		if(SURFACE)
			return m_sSurfaceMap.StorePhoton(p, l, wi, n, &m_uPhotonNumStored);
		else return m_sVolumeMap.StorePhoton(p, l, wi, n, &m_uPhotonNumStored);
	}

	CUDA_ONLY_FUNC float3 L(float a_r, CudaRNG& rng, const e_KernelBSDF* bsdf, const float3& n, const float3& p, const float3& wo) const
	{
		return m_sSurfaceMap.L_Surface(a_r, m_uPhotonNumEmitted, rng, bsdf, n, p, wo);
	}

	CUDA_ONLY_FUNC float3 L(float a_r, CudaRNG& rng, const Ray& r, float tmin, float tmax, const float3& Li) const
	{
		return m_sVolumeMap.L_Volume(a_r, m_uPhotonNumEmitted, rng, r, tmin, tmax, Li);
	}
#endif
};

class k_sPpmTracer : public k_RandTracerBase
{
private:
	k_PhotonMapCollection m_sMaps;
	k_sPpmPixel* m_pDevicePixels;

	float m_fInitialRadius;
	unsigned long long m_uPhotonsEmitted;

	float m_fInitialRadiusScale;
	const unsigned int m_uGridLength;

	float m_uNewPhotonsPerRun;
	int m_uModus;
public:
	k_sPpmTracer();
	virtual ~k_sPpmTracer()
	{
		cudaFree(m_pDevicePixels);
		m_sMaps.Free();
	}
	virtual void Resize(unsigned int _w, unsigned int _h)
	{
		k_TracerBase::Resize(_w, _h);
		if(m_pDevicePixels)
			cudaFree(m_pDevicePixels);
		cudaMalloc(&m_pDevicePixels, w * h * sizeof(k_sPpmPixel));
	}
	virtual void Debug(int2 pixel);
	virtual void PrintStatus(std::vector<FW::String>& a_Buf);
	virtual void CreateSliders(SliderCreateCallback a_Callback);
protected:
	virtual void DoRender(RGBCOL* a_Buf);
	virtual void StartNewTrace(RGBCOL* a_Buf);
private:
	void initNewPass(RGBCOL* a_Buf);
	void doPhotonPass();
	void doEyePass(RGBCOL* a_Buf);
	void updateBuffer()
	{
		unsigned int N = unsigned int(m_uNewPhotonsPerRun * 1000000.0f);
		if(N != m_sMaps.m_uPhotonBufferLength)
			m_sMaps.Resize(N);
	}
	float getCurrentRadius(int exp)
	{
		float f = 1;
		for(int k = 1; k < m_uPassesDone; k++)
			f *= (k + ALPHA) / k;
		f = powf(m_fInitialRadius, float(exp)) * f * 1.0f / float(m_uPassesDone);
		return powf(f, 1.0f / float(exp));
	}
};