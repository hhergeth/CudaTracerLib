#pragma once

#include "k_Tracer.h"
#include "..\Base\CudaRandom.h"
#include "..\Engine\e_PointBVH.h"
#include "..\Base\Timer.h"
#include <time.h>

#define LIGHT_FACTOR (1e7)

struct k_pPpmPhotonEntry
{
	float3 Power;
	float3 Pos;
	float3 Nor;
	float3 Dir;
	CUDA_FUNC_IN k_pPpmPhotonEntry(float3& pos, float3& pow, float3& nor, float3& dir)
	{/*
		Power = half3(pow);
		Pos = half3(pos);
		Nor = NormalizedFloat3ToUchar2(nor);
		Dir = NormalizedFloat3ToUchar2(dir);*/
		Power = pow;
		Pos = pos;
		Nor = nor;
		Dir = dir;
	}
};

class k_pPpmTracer : public k_RandTracerBase
{
private:
	float3* m_pDeviceAccBuffer;
	k_pPpmPhotonEntry* m_pDevicePhotonBuffer;
	AABB m_sEyeBox;
	unsigned int m_uPassIndex;
	unsigned long long m_uNumPhotonsEmitted;
	cTimer m_sTimer;
	float m_fStartRadius, m_fCurrRadius;
	float m_fInitialRadiusScale;
	double m_dTimeRendering, m_dTimeSinceLastUpdate;
	const float m_fAlpha;
	const unsigned int m_uMaxPhotonCount;
public:
	k_pPpmTracer()
		: m_fAlpha(0.7f), m_uMaxPhotonCount(6 * 32 * 180 * 10)
	{
		m_fInitialRadiusScale = 1;
		m_pDeviceAccBuffer = 0;
		cudaMalloc(&m_pDevicePhotonBuffer, m_uMaxPhotonCount * sizeof(k_pPpmPhotonEntry));
	}
	virtual ~k_pPpmTracer()
	{
		cudaFree(m_pDeviceAccBuffer);
		cudaFree(m_pDevicePhotonBuffer);
	}
	virtual void Resize(unsigned int _w, unsigned int _h)
	{
		k_TracerBase::Resize(_w, _h);
		if(m_pDeviceAccBuffer)
			cudaFree(m_pDeviceAccBuffer);
		cudaMalloc(&m_pDeviceAccBuffer, w * h * sizeof(float3));
	}
	virtual void Debug(int2 pixel);
	virtual void PrintStatus(std::vector<FW::String>& a_Buf)
	{
		double pC = floor((double)m_uNumPhotonsEmitted / 1000000.0);
		a_Buf.push_back(FW::sprintf("Photons emitted : %f", (float)pC));
		double pCs = pC / m_dTimeRendering;
		a_Buf.push_back(FW::sprintf("Photons/Sec : %f", (float)pCs));
	}
	virtual void CreateSliders(SliderCreateCallback a_Callback)
	{
		a_Callback(0.01f, 20.0f, true, &m_fInitialRadiusScale, "Initial radius = %g units");
	}
protected:
	virtual void DoRender(RGBCOL* a_Buf);
	virtual void StartNewTrace();
};