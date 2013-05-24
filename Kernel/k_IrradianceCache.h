#pragma once

#include "..\Engine\e_Grid.h"
#include "k_Tracer.h"
#include "..\Base\CudaRandom.h"

struct k_IrrEntry
{
	float3 pos;
	float3 E;
	float3 nor;
	float3 wi;
	float rad;
	unsigned int next;

	CUDA_FUNC_IN k_IrrEntry(const float3& p, const float3& e, const float3& n, float r, unsigned int ne, const float3& _wi)
	{
		pos = p;
		E = e;
		nor = n;
		rad = r;
		next = ne;
		wi = _wi;
	}
};

class k_IrradianceCache : public k_RandTracerBase
{
	k_IrrEntry* m_pEntries;
	unsigned int* m_pGrid;
	k_HashGrid_Irreg m_sGrid;

	const unsigned int m_uEntryNum;
	const unsigned int m_uGridLength;
	float rScale;
public:
	k_IrradianceCache()
		: k_RandTracerBase(), m_uEntryNum(1000000), m_uGridLength(200 * 200 * 200)
	{
		cudaMalloc(&m_pEntries, sizeof(k_IrrEntry) * m_uEntryNum);
		cudaMalloc(&m_pGrid, sizeof(unsigned int) * m_uGridLength);
	}
	virtual ~k_IrradianceCache()
	{
		cudaFree(m_pEntries);
		cudaFree(m_pGrid);
	}
	virtual void Resize(unsigned int _w, unsigned int _h);
	virtual void Debug(int2 pixel);
	virtual void CreateSliders(SliderCreateCallback a_Callback)
	{
		a_Callback(0.1f, 2.0f, false, &rScale, "Radius scale = %g");
	}
protected:
	virtual void DoRender(RGBCOL* a_Buf);
	virtual void StartNewTrace(RGBCOL* a_Buf);
};