#pragma once

#include "k_Tracer.h"
#include "..\Base\CudaRandom.h"
#include "..\Engine\e_PointBVH.h"
#include "..\Base\Timer.h"
#include <time.h>

#define LIGHT_FACTOR (1e1)
//#define USE_BLOCK
#define ALPHA 0.7f
#define GRID_SUBS 200

#define CALC_INDEX(q) (q.z * GRID_SUBS * GRID_SUBS + q.y * GRID_SUBS + q.x)

struct k_sPpmEntry
{
	struct Memory
	{
		unsigned short x;
		unsigned short y;
#ifdef USE_BLOCK
		unsigned int ref;
#endif
		float3 Weight;
	};

	struct Texture
	{
		float3 Pos;
		uchar3 Nor;
		unsigned char DUMMY;
	};

	struct Surface
	{
		unsigned short R2;
		unsigned short N;
		float3 Tau;
	};

	Memory M;
	Texture T;
	Surface S;

	CUDA_FUNC_IN k_sPpmEntry()
	{
	}

	CUDA_FUNC_IN k_sPpmEntry(int _x, int _y, float3& p, float3& n, float3& d, float3& w)
	{
		M.x = (unsigned short)_x;
		M.y = (unsigned short)_y;
#ifdef USE_BLOCK
		M.ref = -1;
#endif
		T.Pos = p;
		float3 qn = normalize(n) * 127.0f + make_float3(127);
		T.Nor = make_uchar3((unsigned char)qn.x, (unsigned char)qn.y, (unsigned char)qn.z);
		S.R2 = -1;
		S.N = 0;
		S.Tau = make_float3(0,0,0);
		M.Weight = w;
	}
};

template<typename T> struct k_PpmBuf
{
public:
	T* m_pHost;
	T* m_pDevice;
	unsigned int Num;
public:
	k_PpmBuf(unsigned int N)
	{
		mallocData(N);
	}
	~k_PpmBuf()
	{
		Free();
	}
	void Resize(unsigned int N)
	{
		Free();
		mallocData(N);
	}
	void Free()
	{
		cudaFree(m_pDevice);
		delete [] m_pHost;
		m_pHost = m_pDevice = 0;
		Num = -1;
	}
	CUDA_HOST void CopyToHost()
	{
		cudaMemcpy(m_pHost, m_pDevice, Num * sizeof(T), cudaMemcpyDeviceToHost);
	}
	CUDA_HOST void CopyToDevice()
	{
		cudaMemcpy(m_pDevice, m_pHost, Num * sizeof(T), cudaMemcpyHostToDevice);
	}
	CUDA_HOST void MemsetDevice(unsigned int val)
	{
		cudaMemset(m_pDevice, val, Num * sizeof(T));
	}
	CUDA_HOST void MemsetHost(unsigned int val)
	{
		memset(m_pHost, val, Num * sizeof(T));
	}
private:
	void mallocData(unsigned int N)
	{
		cudaMalloc(&m_pDevice, sizeof(T) * N);
		m_pHost = new T[N];
		Num = N;
	}
};

template<typename T, typename U> struct k_PpmBufSurf
{
public:
	T* m_pHost;
	cudaArray* m_pDevice;
	T* m_pDeviceTmp;
	unsigned int Num;
public:
	k_PpmBufSurf(unsigned int w, unsigned h)
	{
		mallocData(w,h);
	}
	~k_PpmBufSurf()
	{
		Free();
	}
	void Resize(unsigned int w, unsigned h)
	{
		Free();
		mallocData(w,h);
	}
	void Free()
	{
		cudaFree(m_pDeviceTmp);
		cudaFreeArray(m_pDevice);
		delete [] m_pHost;
		m_pHost = m_pDeviceTmp = 0;
		m_pDevice = 0;
		Num = -1;
	}
	CUDA_HOST void CopyToHost()
	{
		//cudaMemcpyFromArray(m_pHost, m_pDevice, 0, 0, Num * sizeof(T), cudaMemcpyDeviceToHost);
		cudaMemcpy(m_pHost, m_pDeviceTmp, Num * sizeof(T), cudaMemcpyDeviceToHost);
	}
	CUDA_HOST void CopyToDevice()
	{
		cudaMemcpyToArray(m_pDevice, 0, 0, m_pHost, Num * sizeof(T), cudaMemcpyHostToDevice);
	}
	CUDA_HOST void MemsetHost(unsigned int val)
	{
		memset(m_pHost, val, Num * sizeof(T));
	}
	CUDA_HOST void MemsetDevice(unsigned int val)
	{
		cudaMemset(m_pDeviceTmp, val, Num * sizeof(T));
	}
	CUDA_HOST cudaChannelFormatDesc getFormatDesc()
	{
		return cudaCreateChannelDesc<U>();
	}
private:
	void mallocData(unsigned int w, unsigned h)
	{
		cudaChannelFormatDesc d = getFormatDesc();
		cudaMallocArray(&m_pDevice, &d, w, h, cudaArraySurfaceLoadStore);
		cudaMalloc(&m_pDeviceTmp, sizeof(T) * w * h);
		m_pHost = new T[w * h];
		Num = w * h;
	}
};

struct k_sPpmBuffers
{
	k_PpmBuf<k_sPpmEntry::Memory>* m_pBuf0;
	k_PpmBuf<k_sPpmEntry::Texture>* m_pBuf1;
	k_PpmBufSurf<k_sPpmEntry::Surface, float4>* m_pBuf2;

	k_sPpmBuffers(unsigned int w, unsigned int h)
	{
		mallocData(w,h);
	}
	void Resize(unsigned int w, unsigned int h)
	{
		Free();
		mallocData(w,h);
	}
	void Free()
	{
		delete m_pBuf0;
		delete m_pBuf1;
		delete m_pBuf2;
	}
	void MemsetDevice(unsigned int val)
	{
		m_pBuf0->MemsetDevice(val);
		m_pBuf1->MemsetDevice(val);
		m_pBuf2->MemsetDevice(val);
	}
	void CopyToDevice()
	{
		m_pBuf0->CopyToDevice();
		m_pBuf1->CopyToDevice();
		m_pBuf2->CopyToDevice();
	}
	void CopyToHost()
	{
		m_pBuf0->CopyToHost();
		m_pBuf1->CopyToHost();
		m_pBuf2->CopyToHost();
	}
private:
	void mallocData(unsigned int w, unsigned int h)
	{
		m_pBuf0 = new k_PpmBuf<k_sPpmEntry::Memory>(w*h);
		m_pBuf1 = new k_PpmBuf<k_sPpmEntry::Texture>(w*h);
		m_pBuf2 = new k_PpmBufSurf<k_sPpmEntry::Surface, float4>(w, h);
	}
};

struct k_Kernel_sPpmBuffers
{
	k_sPpmEntry::Memory* b0;
	k_sPpmEntry::Texture* b1;
	k_sPpmEntry::Surface* b2;
	unsigned int BufLength;

	k_Kernel_sPpmBuffers(){}

	k_Kernel_sPpmBuffers(k_sPpmBuffers* A, bool a_DeicePointers = true)
	{
		b0 = a_DeicePointers ? A->m_pBuf0->m_pDevice : A->m_pBuf0->m_pHost;
		b1 = a_DeicePointers ? A->m_pBuf1->m_pDevice : A->m_pBuf1->m_pHost;
		b2 = a_DeicePointers ? A->m_pBuf2->m_pDeviceTmp : A->m_pBuf2->m_pHost;
		BufLength = A->m_pBuf0->Num;
	}

	CUDA_HOST void swapEntries(int i, int j)
	{
		k_sPpmEntry::Memory a = b0[i];
		k_sPpmEntry::Texture b = b1[i];
		k_sPpmEntry::Surface c = b2[i];
		b0[i] = b0[j];
		b1[i] = b1[j];
		b2[i] = b2[j];
		b0[j] = a;
		b1[j] = b;
		b2[j] = c;
	}

	CUDA_FUNC_IN void writeEntry(const unsigned int i, const k_sPpmEntry& e)
	{
		b0[i] = e.M;
		b1[i] = e.T;
		b2[i] = e.S;
	}
};

class k_sPpmTracer : public k_TracerBase
{
private:
	k_sPpmBuffers* m_pEyeHits;
	k_PpmBuf<uint2>* m_pGrid;
	k_PpmBuf<curandState>* m_pRngData;
	float3 m_vLow, m_vHigh;
	unsigned long long m_uNumPhotonsEmitted;
	cTimer m_sTimer;
	unsigned int m_uLastValidIndex;
	unsigned int m_uNumEyeHits;
	float m_fStartRadius, m_fCurrRadius;
	double m_dTimeRendering, m_dTimeSinceLastUpdate;
	unsigned int m_uNumMaxEyeHits;
	float m_fInitialRadiusScale, oldRad;
public:
	k_sPpmTracer()
		: k_TracerBase()
	{
		oldRad = m_fInitialRadiusScale = 1;
		m_pEyeHits = 0;
		m_pGrid = new k_PpmBuf<uint2>(GRID_SUBS * GRID_SUBS * GRID_SUBS);
		GenerateRngs();
		m_pEyeHits = new k_sPpmBuffers(1, 1);
	}
	virtual ~k_sPpmTracer()
	{
		delete m_pGrid;
		delete m_pEyeHits;
		delete m_pRngData;
	}
	virtual void Resize(unsigned int _w, unsigned int _h)
	{
		k_TracerBase::Resize(_w, _h);
		m_uNumMaxEyeHits = _w * _h * 2;
		m_pEyeHits->Resize(_w, _h * 2);
	}
	virtual void Debug(int2 pixel);
	virtual void PrintStatus(std::vector<FW::String>& a_Buf)
	{
		double pC = floor((double)m_uNumPhotonsEmitted / 1000000.0);
		a_Buf.push_back(FW::sprintf("Photons emitted : %f", (float)pC));
		double pCs = pC / m_dTimeRendering;
		a_Buf.push_back(FW::sprintf("Photons/Sec : %f", (float)pCs));
	}
	float* getRadiusScalePointer()
	{
		return &m_fInitialRadiusScale;
	}
protected:
	virtual void DoRender(RGBCOL* a_Buf);
	virtual void StartNewTrace();
private:
	void SortHostData();
	void GenerateRngs();
};