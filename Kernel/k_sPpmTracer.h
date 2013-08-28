#pragma once

#include "k_Tracer.h"
#include "..\Base\CudaRandom.h"
#include "..\Engine\e_PointBVH.h"
#include "..\Base\Timer.h"
#include <time.h>
#include "..\Engine\e_Grid.h"

#define ALPHA (2.0f / 3.0f)

struct k_AdaptiveEntry
{
	float r, rd;
	float psi, psi2;
	float I, I2;
	float pl;
};

struct k_AdaptiveStruct
{
	k_AdaptiveEntry* E;
	float r_min;
	float r_max;
	CUDA_FUNC_IN k_AdaptiveStruct(){}
	k_AdaptiveStruct(float rmin, float rmax, k_AdaptiveEntry* e)
	{
		E = e;
		r_min = rmin;
		r_max = rmax;
	}
};

struct k_pPpmPhoton
{
	float3 Pos;
	float3 L;
	float3 Wi;
	float3 Nor;
	unsigned int next;
	CUDA_FUNC_IN k_pPpmPhoton(){}
	CUDA_FUNC_IN k_pPpmPhoton(float3 p, float3 l, float3 wi, float3 n, unsigned int ne)
	{
		Pos = p;
		Nor = n;
		L = (l);
		Wi = wi;
		next = ne;
	}
	CUDA_FUNC_IN float3 getNormal()
	{
		return (Nor);
	}
	CUDA_FUNC_IN float3 getWi()
	{
		return (Wi);
	}
	CUDA_FUNC_IN float3 getL()
	{
		return (L);
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

enum k_StoreResult
{
	Success = 1,
	NotValid = 2,
	Full = 3,
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

	void StartNewRendering(const AABB& box, float a_InitRadius)
	{
		m_sHash = HASH(box, a_InitRadius, m_uGridLength);
		cudaMemset(m_pDeviceHashGrid, -1, sizeof(unsigned int) * m_uGridLength);
	}

#ifdef __CUDACC__
	CUDA_ONLY_FUNC k_StoreResult StorePhoton(const float3& p, const float3& l, const float3& wi, const float3& n, unsigned int* a_PhotonCounter) const
	{
		if(!m_sHash.IsValidHash(p))
			return k_StoreResult::NotValid;
		uint3 i0 = m_sHash.Transform(p);
		unsigned int i = m_sHash.Hash(i0);
		unsigned int j = atomicInc(a_PhotonCounter, -1);
		if(j < m_uMaxPhotonCount)
		{
			unsigned int k = atomicExch(m_pDeviceHashGrid + i, j);
			m_pDevicePhotons[j] = k_pPpmPhoton(p, l, wi, n, k);//m_sHash.EncodePos(p, i0)
			return k_StoreResult::Success;
		}
		return k_StoreResult::Full;
	}

	CUDA_ONLY_FUNC float3 L_Surface(float a_r, float a_NumPhotonEmitted, CudaRNG& rng, const e_KernelBSDF* bsdf, const float3& n, const float3& p, const float3& wo, k_AdaptiveStruct& A) const;

	template<bool VOL> CUDA_ONLY_FUNC float3 L_Volume(float a_r, float a_NumPhotonEmitted, CudaRNG& rng, const Ray& r, float tmin, float tmax, const float3& sigt) const;
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

	void StartNewRendering(const AABB& sbox, const AABB& vbox, float a_R)
	{
		m_sVolumeMap.StartNewRendering(vbox, a_R);
		m_sSurfaceMap.StartNewRendering(sbox, a_R);
	}

#ifdef __CUDACC__

	template<bool SURFACE> CUDA_ONLY_FUNC k_StoreResult StorePhoton(const float3& p, const float3& l, const float3& wi, const float3& n)
	{
		if(SURFACE)
			return m_sSurfaceMap.StorePhoton(p, l, wi, n, &m_uPhotonNumStored);
		else return m_sVolumeMap.StorePhoton(p, l, wi, n, &m_uPhotonNumStored);
	}

	CUDA_ONLY_FUNC float3 L(float a_r, CudaRNG& rng, const e_KernelBSDF* bsdf, const float3& n, const float3& p, const float3& wo, k_AdaptiveStruct& A) const
	{
		return m_sSurfaceMap.L_Surface(a_r, m_uPhotonNumEmitted, rng, bsdf, n, p, wo, A);
	}

	template<bool VOL> CUDA_ONLY_FUNC float3 L(float a_r, CudaRNG& rng, const Ray& r, float tmin, float tmax, const float3& sigt) const
	{
		return m_sVolumeMap.L_Volume<VOL>(a_r, m_uPhotonNumEmitted, rng, r, tmin, tmax, sigt);
	}
#endif
};

class k_sPpmTracer : public k_ProgressiveTracer
{
private:
	k_PhotonMapCollection m_sMaps;
	bool m_bDirect;

	float m_fInitialRadius;
	unsigned long long m_uPhotonsEmitted;

	float m_fInitialRadiusScale;
	const unsigned int m_uGridLength;

	float m_uNewPhotonsPerRun;
	int m_uModus;
	bool m_bLongRunning;

	k_AdaptiveEntry* m_pEntries;
	float r_min, r_max;
public:
	k_sPpmTracer();
	virtual ~k_sPpmTracer()
	{
		m_sMaps.Free();
	}
	virtual void Resize(unsigned int _w, unsigned int _h);
	virtual void Debug(int2 pixel);
	virtual void PrintStatus(std::vector<FW::String>& a_Buf);
	virtual void CreateSliders(SliderCreateCallback a_Callback);
protected:
	virtual void DoRender(e_Image* I);
	virtual void StartNewTrace(e_Image* I);
private:
	void initNewPass(e_Image* I);
	void doPhotonPass();
	void doEyePass(e_Image* I);
	void doStartPass(float r, float rd);
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