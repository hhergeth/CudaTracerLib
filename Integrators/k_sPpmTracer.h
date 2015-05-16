#pragma once

#include "..\Kernel\k_Tracer.h"
#include "..\Base\CudaRandom.h"
#include "..\Base\Timer.h"
#include <time.h>
#include "k_PhotonMapHelper.h"

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

class k_sPpmTracer : public k_Tracer<true, true>
{
private:
	k_PhotonMapCollection<true, k_pPpmPhoton> m_sMaps;
	bool m_bDirect;
	float m_fLightVisibility;

	float m_fInitialRadius;
	unsigned long long m_uPhotonsEmitted;

	unsigned int m_uBlocksPerLaunch;

	bool m_bLongRunning;

	k_AdaptiveEntry* m_pEntries;
	float r_min, r_max;
public:
	bool m_bVisualizeGrid;
	unsigned int m_uVisLastMax;
	bool m_bFinalGather;
	k_sPpmTracer();
	virtual ~k_sPpmTracer()
	{
		m_sMaps.Free();
		CUDA_FREE(m_pEntries);
	}
	virtual void Resize(unsigned int _w, unsigned int _h);
	virtual void Debug(e_Image* I, const Vec2i& pixel);
	virtual void PrintStatus(std::vector<std::string>& a_Buf) const;
	virtual void CreateSliders(SliderCreateCallback a_Callback) const;
	CUDA_FUNC_IN static float getCurrentRadius(float initial_r, unsigned int iteration, float exp)
	{
		return math::pow(math::pow(initial_r, exp) / math::pow(float(iteration), 0.5f * (1 - ALPHA)), 1.0f / exp);
	}
	float getCurrentRadius2(float exp)
	{
		return getCurrentRadius(m_fInitialRadius, m_uPassesDone, exp);
	}
protected:
	virtual void DoRender(e_Image* I);
	virtual void StartNewTrace(e_Image* I);
	virtual void RenderBlock(e_Image* I, int x, int y, int blockW, int blockH);
private:
	void doPhotonPass();
	void doStartPass(float r, float rd);
	void estimatePerPixelRadius();
	void visualizeGrid(e_Image* I);
};