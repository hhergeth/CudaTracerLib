#pragma once

#include <Kernel/Tracer.h>
#include <Engine/Grid.h>
#include <Math/Compression.h>
#include <Base/FileStream.h>
#include <CudaMemoryManager.h>

namespace CudaTracerLib {

#define ALPHA (2.0f / 3.0f)

CUDA_FUNC_IN float getCurrentRadius(float initial_r, unsigned int iteration, float exp)
{
	return initial_r * math::pow(iteration, (ALPHA - 1) / exp);
}

//smoothing kernel which integrates [-1, +1] to 1
CUDA_FUNC_IN float k(float t)
{
	t = math::clamp01(t);
	return (1.0f + t * t * t * (-6.0f * t * t + 15.0f * t - 10.0f));
}

CUDA_FUNC_IN float k_tr(float r, float t)
{
	return k(t / r) / (r * r);
}

template<typename VEC> CUDA_FUNC_IN float k_tr(float r, const VEC& t)
{
	return k_tr(r, length(t));
}

enum PhotonType
{
	pt_Diffuse = 0,
	pt_Volume = 1,
	pt_Caustic = 2,
};

struct PPPMPhoton
{
private:
	RGBE L;
	unsigned short Wi;
	unsigned short Nor;
	unsigned int typeNextField;
public:
	Vec3f Pos;
	CUDA_FUNC_IN PPPMPhoton(){}
	CUDA_FUNC_IN PPPMPhoton(const Spectrum& l, const Vec3f& wi, const Vec3f& n, PhotonType type)
	{
		Nor = NormalizedFloat3ToUchar2(n);
		L = (l).toRGBE();
		Wi = NormalizedFloat3ToUchar2(wi);
		typeNextField = ((unsigned char)type << 24) | 0xffffff;
	}
	CUDA_FUNC_IN bool hasNext() const
	{
		return (typeNextField & 0xffffff) != 0xffffff;
	}
	CUDA_FUNC_IN PhotonType getType() const
	{
		return PhotonType(typeNextField >> 24);
	}
	CUDA_FUNC_IN unsigned int getNext() const
	{
		return typeNextField & 0xffffff;
	}
	CUDA_FUNC_IN void setNext(unsigned int next)
	{
		typeNextField = (typeNextField & 0xff000000) | (next & 0xffffff);
	}
	CUDA_FUNC_IN Vec3f getNormal() const
	{
		return Uchar2ToNormalizedFloat3(Nor);
	}
	CUDA_FUNC_IN Vec3f getWi() const
	{
		return Uchar2ToNormalizedFloat3(Wi);
	}
	CUDA_FUNC_IN Spectrum getL() const
	{
		Spectrum s;
		s.fromRGBE(L);
		return s;
	}
	template<typename HASH> CUDA_FUNC_IN void setPos(const HASH& hash, const Vec3u& i, const Vec3f& p)
	{
		//pos = hash.EncodePos(p, i);
		Pos = p;
	}
	template<typename HASH> CUDA_FUNC_IN Vec3f getPos(const HASH& hash, const Vec3u& i) const
	{
		//return hash.DecodePos(pos, i);
		return Pos;
	}
	CUDA_FUNC_IN unsigned short& accessNormalStorage()
	{
		return Nor;
	}
	CUDA_FUNC_IN bool getFlag() const
	{
		return (bool)(typeNextField >> 31);
	}
	CUDA_FUNC_IN void setFlag(bool b)
	{
		typeNextField |= (unsigned int)b << 31;
	}
	/*
	 CUDA_FUNC_IN PPPMPhoton(const uint4& v)
	 {
	 Pos = v.x;
	 L = *(RGBE*)&v.y;
	 Wi = make_uchar2(v.z & 0xff, (v.z >> 8) & 0xff);
	 Nor = make_uchar2((v.z >> 16) & 0xff, v.z >> 24);
	 next = v.w;
	 }*/
};

struct k_MISPhoton : public PPPMPhoton
{
	float dVC, dVCM, dVM;
	CUDA_FUNC_IN k_MISPhoton(){}
	CUDA_FUNC_IN k_MISPhoton(const Spectrum& l, const Vec3f& wi, const Vec3f& n, PhotonType type)
		: PPPMPhoton(l, wi, n, type)
	{
	}
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

	void Serialize(FileOutputStream& O, void* hostbuf)
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
		m_sHash = HASH(box, (unsigned int)math::pow(m_uGridLength, 1.0f / 3.0f));
		ThrowCudaErrors(cudaMemset(m_pDeviceHashGrid, -1, sizeof(unsigned int)* m_uGridLength));
		//ThrowCudaErrors(cudaMemset(m_pDeviceLinkedList, -1, sizeof(uint2)* m_uLinkedListLength));
		//m_uLinkedListUsed = 0;
	}
};

template<bool HAS_MULTIPLE_MAPS, typename PHOTON> struct k_PhotonMapCollection
{
	k_PhotonMap<HashGrid_Reg> m_sSurfaceMap;
	k_PhotonMap<HashGrid_Reg> m_sVolumeMap;
	k_PhotonMap<HashGrid_Reg> m_sCausticMap;
	PHOTON* m_pPhotons;// , *m_pPhotons2;
	unsigned int m_uPhotonBufferLength;
	unsigned int m_uPhotonNumStored;
	unsigned int m_uPhotonNumEmitted;

	CUDA_FUNC_IN k_PhotonMapCollection()
	{

	}

	k_PhotonMapCollection(unsigned int a_BufferLength, unsigned int a_HashNum, unsigned int linkedListLength)
		: m_pPhotons(0)
	{
		m_sSurfaceMap = k_PhotonMap<HashGrid_Reg>(a_HashNum);
		if (HAS_MULTIPLE_MAPS)
		{
			m_sVolumeMap = k_PhotonMap<HashGrid_Reg>(a_HashNum);
			m_sCausticMap = k_PhotonMap<HashGrid_Reg>(a_HashNum);
		}
		m_uPhotonBufferLength = 0;
		m_uPhotonNumStored = m_uPhotonNumEmitted = 0;
		Resize(a_BufferLength, linkedListLength);
	}

	void Serialize(FileOutputStream& O)
	{
		void* hostbuf = malloc(m_uPhotonBufferLength * sizeof(PHOTON));
		O << m_uPhotonBufferLength;
		O << m_uPhotonNumStored;
		O << m_uPhotonNumEmitted;
		cudaMemcpy(hostbuf, m_pPhotons, m_uPhotonBufferLength * sizeof(PHOTON), cudaMemcpyDeviceToHost);
		O.Write(hostbuf, m_uPhotonBufferLength * sizeof(PHOTON));
		m_sSurfaceMap.Serialize(O, hostbuf);
		m_sVolumeMap.Serialize(O, hostbuf);
		m_sCausticMap.Serialize(O, hostbuf);
		free(hostbuf);
	}

	void DeSerialize(FileInputStream& I)
	{
		void* hostbuf = malloc(m_uPhotonBufferLength * sizeof(PHOTON));
		I >> m_uPhotonBufferLength;
		I >> m_uPhotonNumStored;
		I >> m_uPhotonNumEmitted;
		I.Read(hostbuf, m_uPhotonBufferLength * sizeof(PHOTON));
		cudaMemcpy(m_pPhotons, hostbuf, m_uPhotonBufferLength * sizeof(PHOTON), cudaMemcpyHostToDevice);
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
		CUDA_MALLOC(&m_pPhotons, sizeof(PHOTON)* a_BufferLength);
		cudaMemset(m_pPhotons, 0, sizeof(PHOTON)* a_BufferLength);
		//if (m_pPhotons2)
		//	CUDA_FREE(m_pPhotons2);
		//CUDA_MALLOC(&m_pPhotons2, sizeof(PHOTON)* a_BufferLength);
		//cudaMemset(m_pPhotons2, 0, sizeof(PHOTON)* a_BufferLength);
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
		cudaMemset(m_pPhotons, 0, sizeof(PHOTON)* m_uPhotonBufferLength);
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

typedef k_PhotonMap<HashGrid_Reg> k_PhotonMapReg;

#ifdef __CUDACC__
template<typename PHOTON> CUDA_ONLY_FUNC bool storePhoton(const Vec3f& pos, const Spectrum& phi, const Vec3f& wi, const Vec3f& n, PhotonType type, k_PhotonMapCollection<true, PHOTON>& g_Map, bool final_gather)
{
	unsigned int p_idx = atomicInc(&g_Map.m_uPhotonNumStored, UINT_MAX);
	if (p_idx < g_Map.m_uPhotonBufferLength)
	{
		{
			PHOTON p = PHOTON(phi, wi, n, type);
			const k_PhotonMap<HashGrid_Reg>& map = type == pt_Volume ? g_Map.m_sVolumeMap :  g_Map.m_sSurfaceMap;
			p.setPos(map.m_sHash, map.m_sHash.Transform(pos), pos);
			unsigned int grid_idx = map.m_sHash.Hash(pos);
			unsigned int k = atomicExch(map.m_pDeviceHashGrid + grid_idx, p_idx);
			p.setNext(k);
			g_Map.m_pPhotons[p_idx] = p;
		}
		//if this photon is caustic we will also have to store it in the diffuse map
		if (type == PhotonType::pt_Caustic && final_gather)
		{
			p_idx = atomicInc(&g_Map.m_uPhotonNumStored, UINT_MAX);
			if (p_idx < g_Map.m_uPhotonBufferLength)
			{
				PHOTON p = PHOTON(phi, wi, n, PhotonType::pt_Diffuse);
				const k_PhotonMap<HashGrid_Reg>& map = g_Map.m_sCausticMap;
				p.setPos(map.m_sHash, map.m_sHash.Transform(pos), pos);
				unsigned int grid_idx = map.m_sHash.Hash(pos);
				unsigned int k = atomicExch(map.m_pDeviceHashGrid + grid_idx, p_idx);
				p.setNext(k);
				g_Map.m_pPhotons[p_idx] = p;
			}
		}
		return true;
	}
	else return false;
}

template<typename PHOTON> CUDA_ONLY_FUNC bool storePhoton(const Vec3f& pos, const Spectrum& phi, const Vec3f& wi, const Vec3f& n, PhotonType type, k_PhotonMapCollection<false, PHOTON>& g_Map, PHOTON** resPhoton = 0)
{
	unsigned int p_idx = atomicInc(&g_Map.m_uPhotonNumStored, UINT_MAX);
	if (p_idx < g_Map.m_uPhotonBufferLength)
	{
		PHOTON p = PHOTON(phi, wi, n, type);
		const k_PhotonMap<HashGrid_Reg>& map = type == pt_Volume ? g_Map.m_sVolumeMap :  g_Map.m_sSurfaceMap;
		p.setPos(map.m_sHash, map.m_sHash.Transform(pos), pos);
		unsigned int grid_idx = map.m_sHash.Hash(pos);
		unsigned int k = atomicExch(map.m_pDeviceHashGrid + grid_idx, p_idx);
		p.setNext(k);
		g_Map.m_pPhotons[p_idx] = p;
		if (resPhoton)
			*resPhoton = g_Map.m_pPhotons + p_idx;
		return true;
	}
	else return false;
}
#endif

}
