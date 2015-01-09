#pragma once

#include "..\Kernel\k_Tracer.h"
#include "..\Base\CudaRandom.h"
#include "..\Base\Timer.h"
#include <time.h>
#include "..\Engine\e_Grid.h"
#include "../Math//Compression.h"

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

enum PhotonType
{
	pt_Diffuse = 0,
	pt_Volume = 1,
	pt_Caustic = 2,
};

struct k_pPpmPhoton
{
private:
	float3 Pos;
	//Spectrum L;
	//float3 Wi;
	//float3 Nor;
	RGBE L;
	unsigned short Wi;
	unsigned short Nor;
	unsigned int typeNextField;
public:
	CUDA_FUNC_IN k_pPpmPhoton(){}
	CUDA_FUNC_IN k_pPpmPhoton(const float3& p, const Spectrum& l, const float3& wi, const float3& n, PhotonType type)
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
	CUDA_FUNC_IN float3 getNormal()
	{
		return Uchar2ToNormalizedFloat3(Nor);
	}
	CUDA_FUNC_IN float3 getWi()
	{
		return Uchar2ToNormalizedFloat3(Wi);
	}
	CUDA_FUNC_IN Spectrum getL()
	{
		Spectrum s;
		s.fromRGBE(L);
		return s;
	}
	CUDA_FUNC_IN float3 getPos()
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

	CUDA_FUNC_IN k_PhotonMap(unsigned int hashN)
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
		cudaMemcpy(hostbuf, m_pDeviceHashGrid, sizeof(unsigned int) * m_uGridLength, cudaMemcpyDeviceToHost);
		O.Write(hostbuf, sizeof(unsigned int) * m_uGridLength);
		O.Write(m_sHash);
	}

	void DeSerialize(IInStream& I, void* hostbuf)
	{
		I >> m_uGridLength;
		I.Read(hostbuf, sizeof(unsigned int) * m_uGridLength);
		cudaMemcpy(m_pDeviceHashGrid, hostbuf, sizeof(unsigned int) * m_uGridLength, cudaMemcpyHostToDevice);
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

struct k_PhotonMapCollection
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
		m_sVolumeMap = k_PhotonMap<k_HashGrid_Reg>(a_HashNum);
		m_sCausticMap = k_PhotonMap<k_HashGrid_Reg>(a_HashNum);
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
		m_sVolumeMap.Free();
		m_sCausticMap.Free();
		CUDA_FREE(m_pPhotons);
	}

	void Resize(unsigned int a_BufferLength, unsigned int linkedListLength)
	{
		CUDA_MALLOC(&m_pPhotons, sizeof(k_pPpmPhoton)* a_BufferLength);
		cudaMemset(m_pPhotons, 0, sizeof(k_pPpmPhoton)* a_BufferLength);
		m_uPhotonBufferLength = a_BufferLength;
		m_sSurfaceMap.Resize(linkedListLength);
		m_sVolumeMap.Resize(linkedListLength);
		m_sCausticMap.Resize(linkedListLength);
	}

	void StartNewPass()
	{
		m_uPhotonNumEmitted = m_uPhotonNumStored = 0;
		m_sSurfaceMap.StartNewPass();
		m_sVolumeMap.StartNewPass();
		m_sCausticMap.StartNewPass();
		cudaMemset(m_pPhotons, 0, sizeof(k_pPpmPhoton)* m_uPhotonBufferLength);
	}

	bool PassFinished()
	{
		return m_uPhotonNumStored >= m_uPhotonBufferLength;
	}

	void StartNewRendering(const AABB& surfaceBox, const AABB& volBox, float a_R)
	{
		m_sSurfaceMap.StartNewRendering(surfaceBox, a_R);
		m_sVolumeMap.StartNewRendering(volBox, a_R);
		m_sCausticMap.StartNewRendering(surfaceBox, a_R);
	}
};

typedef k_PhotonMap<k_HashGrid_Reg> k_PhotonMapReg;

enum
{
	PPM_Photons_Per_Thread = 12,
	PPM_BlockX = 32,
	PPM_BlockY = 6,
	PPM_MaxRecursion = 6,

	PPM_photons_per_block = PPM_Photons_Per_Thread * PPM_BlockX * PPM_BlockY,
	PPM_slots_per_thread = PPM_Photons_Per_Thread * PPM_MaxRecursion,
	PPM_slots_per_block = PPM_photons_per_block * PPM_MaxRecursion,
};

class k_sPpmTracer : public k_ProgressiveTracer
{
private:
	k_PhotonMapCollection m_sMaps;
	bool m_bDirect;
	float m_fLightVisibility;

	float m_fInitialRadius;
	unsigned long long m_uPhotonsEmitted;
	unsigned long long m_uPreviosCount;

	const unsigned int m_uGridLength;

	unsigned int m_uBlocksPerLaunch;

	int m_uModus;
	bool m_bLongRunning;

	k_AdaptiveEntry* m_pEntries;
	float r_min, r_max;
public:
	bool m_bFinalGather;
	k_sPpmTracer();
	virtual ~k_sPpmTracer()
	{
		m_sMaps.Free();
	}
	virtual void Resize(unsigned int _w, unsigned int _h);
	virtual void Debug(e_Image* I, int2 pixel);
	virtual void PrintStatus(std::vector<std::string>& a_Buf);
	virtual void CreateSliders(SliderCreateCallback a_Callback);
protected:
	virtual void DoRender(e_Image* I);
	virtual void StartNewTrace(e_Image* I);
private:
	void initNewPass(e_Image* I);
	void doPhotonPass();
	void doEyePass(e_Image* I);
	void doStartPass(float r, float rd);
	float getCurrentRadius(int exp)
	{
		float f = 1;
		for(unsigned int k = 1; k < m_uPassesDone; k++)
			f *= (k + ALPHA) / k;
		f = powf(m_fInitialRadius, float(exp)) * f * 1.0f / float(m_uPassesDone);
		return powf(f, 1.0f / float(exp));
	}
};