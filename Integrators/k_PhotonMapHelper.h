#pragma once

#include "..\Kernel\k_Tracer.h"
#include "..\Base\CudaRandom.h"
#include "..\Engine\e_Grid.h"
#include "../Math//Compression.h"

#define ALPHA (2.0f / 3.0f)

CUDA_FUNC_IN float k(float t)
{
	//float t2 = t * t;
	//return 1.0f + t2 * t * (-6.0f * t2 + 15.0f * t - 10.0f);
	return math::clamp01(1.0f + t * t * t * (-6.0f * t * t + 15.0f * t - 10.0f));
}

CUDA_FUNC_IN float k_tr(float r, float t)
{
	//if (t > r)
	//	printf("t : %f, r : %f", t, r);
	return k(t / r) / (PI * r * r);
}

CUDA_FUNC_IN float k_tr(float r, const Vec3f& t)
{
	return k_tr(r, length(t));
}


enum PhotonType
{
	pt_Diffuse = 0,
	pt_Volume = 1,
	pt_Caustic = 2,
};

struct k_pPpmPhoton
{
private:
	Vec3f Pos;
	//Spectrum L;
	//float3 Wi;
	//float3 Nor;
	RGBE L;
	unsigned short Wi;
	unsigned short Nor;
	unsigned int typeNextField;
public:
	float dVC, dVCM, dVM;
	CUDA_FUNC_IN k_pPpmPhoton(){}
	CUDA_FUNC_IN k_pPpmPhoton(const Vec3f& p, const Spectrum& l, const Vec3f& wi, const Vec3f& n, PhotonType type)
	{
		Pos = p;
		Nor = NormalizedFloat3ToUchar2(n);
		L = (l).toRGBE();
		Wi = NormalizedFloat3ToUchar2(wi);
		typeNextField = (unsigned char(type) << 24) | 0xffffff;
	}
	CUDA_FUNC_IN bool hasNext()
	{
		return (typeNextField & 0xffffff) != 0xffffff;
	}
	CUDA_FUNC_IN PhotonType getType()
	{
		return PhotonType(typeNextField >> 24);
	}
	CUDA_FUNC_IN unsigned int getNext()
	{
		return typeNextField & 0xffffff;
	}
	CUDA_FUNC_IN void setNext(unsigned int next)
	{
		typeNextField = (typeNextField & 0xff000000) | (next & 0xffffff);
	}
	CUDA_FUNC_IN Vec3f getNormal()
	{
		return Uchar2ToNormalizedFloat3(Nor);
	}
	CUDA_FUNC_IN Vec3f getWi()
	{
		return Uchar2ToNormalizedFloat3(Wi);
	}
	CUDA_FUNC_IN Spectrum getL()
	{
		Spectrum s;
		s.fromRGBE(L);
		return s;
	}
	CUDA_FUNC_IN Vec3f getPos()
	{/*
	 uint4 r;
	 r.x = Pos;
	 r.y = (L.x << 24) | (L.y << 16) | (L.z << 8) | L.w;
	 r.z = (L.x << 24) | (L.y << 16) | (L.z << 8) | L.w;
	 r.w = next;
	 return r;*/
		return Pos;
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

template<typename HASH> struct k_PhotonMap
{
	unsigned int* m_pDeviceHashGrid;
	//uint2* m_pDeviceLinkedList;
	HASH m_sHash;
	unsigned int m_uGridLength;
	//unsigned int m_uLinkedListLength;
	//unsigned int m_uLinkedListUsed;

	CUDA_FUNC_IN k_PhotonMap()
	{

	}

	k_PhotonMap(unsigned int hashN)
	{
		m_uGridLength = hashN;
		m_pDeviceHashGrid = 0;
		//m_pDeviceLinkedList = 0;
		CUDA_MALLOC(&m_pDeviceHashGrid, sizeof(unsigned int)* m_uGridLength);
		ThrowCudaErrors(cudaMemset(m_pDeviceHashGrid, -1, sizeof(unsigned int)* m_uGridLength));
	}

	void Resize(unsigned int linkedListLength)
	{
		//m_uLinkedListUsed = 0;
		//m_uLinkedListLength = linkedListLength;
		//if (m_pDeviceLinkedList)
		//	CUDA_FREE(m_pDeviceLinkedList);
		//CUDA_MALLOC(&m_pDeviceLinkedList, sizeof(uint2) * linkedListLength);
		//ThrowCudaErrors(cudaMemset(m_pDeviceLinkedList, -1, sizeof(uint2) * linkedListLength));
	}

	void Serialize(OutputStream& O, void* hostbuf)
	{
		O << m_uGridLength;
		cudaMemcpy(hostbuf, m_pDeviceHashGrid, sizeof(unsigned int)* m_uGridLength, cudaMemcpyDeviceToHost);
		O.Write(hostbuf, sizeof(unsigned int)* m_uGridLength);
		O.Write(m_sHash);
	}

	void DeSerialize(IInStream& I, void* hostbuf)
	{
		I >> m_uGridLength;
		I.Read(hostbuf, sizeof(unsigned int)* m_uGridLength);
		cudaMemcpy(m_pDeviceHashGrid, hostbuf, sizeof(unsigned int)* m_uGridLength, cudaMemcpyHostToDevice);
		I.Read(m_sHash);
	}

	void Free()
	{
		CUDA_FREE(m_pDeviceHashGrid);
		//CUDA_FREE(m_pDeviceLinkedList);
	}

	void StartNewPass()
	{
		ThrowCudaErrors(cudaMemset(m_pDeviceHashGrid, -1, sizeof(unsigned int)* m_uGridLength));
		//ThrowCudaErrors(cudaMemset(m_pDeviceLinkedList, -1, sizeof(uint2)* m_uLinkedListLength));
		//m_uLinkedListUsed = 0;
	}

	void StartNewRendering(const AABB& box, float a_InitRadius)
	{
		m_sHash = HASH(box, a_InitRadius, m_uGridLength);
		ThrowCudaErrors(cudaMemset(m_pDeviceHashGrid, -1, sizeof(unsigned int)* m_uGridLength));
		//ThrowCudaErrors(cudaMemset(m_pDeviceLinkedList, -1, sizeof(uint2)* m_uLinkedListLength));
		//m_uLinkedListUsed = 0;
	}
};

template<bool HAS_MULTIPLE_MAPS> struct k_PhotonMapCollection
{
	k_PhotonMap<k_HashGrid_Reg> m_sSurfaceMap;
	k_PhotonMap<k_HashGrid_Reg> m_sVolumeMap;
	k_PhotonMap<k_HashGrid_Reg> m_sCausticMap;
	k_pPpmPhoton* m_pPhotons;
	unsigned int m_uPhotonBufferLength;
	unsigned int m_uPhotonNumStored;
	unsigned int m_uPhotonNumEmitted;

	CUDA_FUNC_IN k_PhotonMapCollection()
	{

	}

	k_PhotonMapCollection(unsigned int a_BufferLength, unsigned int a_HashNum, unsigned int linkedListLength)
		: m_pPhotons(0)
	{
		m_sSurfaceMap = k_PhotonMap<k_HashGrid_Reg>(a_HashNum);
		if (HAS_MULTIPLE_MAPS)
		{
			m_sVolumeMap = k_PhotonMap<k_HashGrid_Reg>(a_HashNum);
			m_sCausticMap = k_PhotonMap<k_HashGrid_Reg>(a_HashNum);
		}
		m_uPhotonBufferLength = 0;
		m_uPhotonNumStored = m_uPhotonNumEmitted = 0;
		Resize(a_BufferLength, linkedListLength);
	}

	void Serialize(OutputStream& O)
	{
		void* hostbuf = malloc(m_uPhotonBufferLength * sizeof(k_pPpmPhoton));
		O << m_uPhotonBufferLength;
		O << m_uPhotonNumStored;
		O << m_uPhotonNumEmitted;
		cudaMemcpy(hostbuf, m_pPhotons, m_uPhotonBufferLength * sizeof(k_pPpmPhoton), cudaMemcpyDeviceToHost);
		O.Write(hostbuf, m_uPhotonBufferLength * sizeof(k_pPpmPhoton));
		m_sSurfaceMap.Serialize(O, hostbuf);
		m_sVolumeMap.Serialize(O, hostbuf);
		m_sCausticMap.Serialize(O, hostbuf);
		free(hostbuf);
	}

	void DeSerialize(InputStream& I)
	{
		void* hostbuf = malloc(m_uPhotonBufferLength * sizeof(k_pPpmPhoton));
		I >> m_uPhotonBufferLength;
		I >> m_uPhotonNumStored;
		I >> m_uPhotonNumEmitted;
		I.Read(hostbuf, m_uPhotonBufferLength * sizeof(k_pPpmPhoton));
		cudaMemcpy(m_pPhotons, hostbuf, m_uPhotonBufferLength * sizeof(k_pPpmPhoton), cudaMemcpyHostToDevice);
		m_sSurfaceMap.DeSerialize(I, hostbuf);
		m_sVolumeMap.DeSerialize(I, hostbuf);
		m_sCausticMap.DeSerialize(I, hostbuf);
		free(hostbuf);
	}

	void Free()
	{
		m_sSurfaceMap.Free();
		if (HAS_MULTIPLE_MAPS)
		{
			m_sVolumeMap.Free();
			m_sCausticMap.Free();
		}
		CUDA_FREE(m_pPhotons);
	}

	void Resize(unsigned int a_BufferLength, unsigned int linkedListLength)
	{
		if (m_pPhotons)
			CUDA_FREE(m_pPhotons);
		CUDA_MALLOC(&m_pPhotons, sizeof(k_pPpmPhoton)* a_BufferLength);
		cudaMemset(m_pPhotons, 0, sizeof(k_pPpmPhoton)* a_BufferLength);
		m_uPhotonBufferLength = a_BufferLength;
		m_sSurfaceMap.Resize(linkedListLength);
		if (HAS_MULTIPLE_MAPS)
		{
			m_sVolumeMap.Resize(linkedListLength);
			m_sCausticMap.Resize(linkedListLength);
		}
	}

	void StartNewPass()
	{
		m_uPhotonNumEmitted = m_uPhotonNumStored = 0;
		m_sSurfaceMap.StartNewPass();
		if (HAS_MULTIPLE_MAPS)
		{
			m_sVolumeMap.StartNewPass();
			m_sCausticMap.StartNewPass();
		}
		cudaMemset(m_pPhotons, 0, sizeof(k_pPpmPhoton)* m_uPhotonBufferLength);
	}

	bool PassFinished()
	{
		return m_uPhotonNumStored >= m_uPhotonBufferLength;
	}

	void StartNewRendering(const AABB& surfaceBox, const AABB& volBox, float a_R)
	{
		m_sSurfaceMap.StartNewRendering(surfaceBox, a_R);
		if (HAS_MULTIPLE_MAPS)
		{
			m_sVolumeMap.StartNewRendering(volBox, a_R);
			m_sCausticMap.StartNewRendering(surfaceBox, a_R);
		}
	}
};

typedef k_PhotonMap<k_HashGrid_Reg> k_PhotonMapReg;

#ifdef __CUDACC__
CUDA_ONLY_FUNC bool storePhoton(const Vec3f& p, const Spectrum& phi, const Vec3f& wi, const Vec3f& n, PhotonType type, k_PhotonMapCollection<true>& g_Map)
{
	unsigned int p_idx = atomicInc(&g_Map.m_uPhotonNumStored, 0xffffffff);
	if (p_idx < g_Map.m_uPhotonBufferLength)
	{
		g_Map.m_pPhotons[p_idx] = k_pPpmPhoton(p, phi, wi, n, type);
		//if this photon is caustic we will also have to store it in the diffuse map
		if (type == PhotonType::pt_Caustic)
		{
			p_idx = atomicInc(&g_Map.m_uPhotonNumStored, 0xffffffff);
			if (p_idx < g_Map.m_uPhotonBufferLength)
			{
				g_Map.m_pPhotons[p_idx] = k_pPpmPhoton(p, phi, wi, n, PhotonType::pt_Diffuse);
			}
		}
		return true;
	}
	else return false;
}

CUDA_ONLY_FUNC bool storePhoton(const Vec3f& p, const Spectrum& phi, const Vec3f& wi, const Vec3f& n, PhotonType type, k_PhotonMapCollection<false>& g_Map, k_pPpmPhoton** resPhoton = 0)
{
	unsigned int p_idx = atomicInc(&g_Map.m_uPhotonNumStored, 0xffffffff);
	if (p_idx < g_Map.m_uPhotonBufferLength)
	{
		g_Map.m_pPhotons[p_idx] = k_pPpmPhoton(p, phi, wi, n, type);
		if (resPhoton)
			*resPhoton = g_Map.m_pPhotons + p_idx;
		return true;
	}
	else return false;
}
#endif