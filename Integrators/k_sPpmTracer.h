#pragma once

#include <Kernel/k_Tracer.h>
#include "k_PhotonMapHelper.h"
#include <CudaMemoryManager.h>
#include "VolEstimators/k_BeamBeamGrid.h"
#include "VolEstimators/k_BeamBVHStorage.h"
#include "VolEstimators/k_BeamGrid.h"

namespace CudaTracerLib {

struct k_AdaptiveEntry
{
	float r, rd;
	float psi, psi2;
	float I, I2;
	float pl;
	int n1, n2;
};

struct k_AdaptiveStruct
{
	float r_min;
	float r_max;
	int w;
	CUDA_FUNC_IN k_AdaptiveStruct(){}
	k_AdaptiveStruct(float rmin, float rmax, k_AdaptiveEntry* e, int w, int m_uPassesDone)
		: w(w)
	{
		E = e;
		r_min = rmin * math::pow(float(m_uPassesDone), -1.0f / 6.0f);
		r_max = rmax * math::pow(float(m_uPassesDone), -1.0f / 6.0f);
	}
	CUDA_FUNC_IN k_AdaptiveEntry& operator()(int x, int y)
	{
		return E[w * y + x];
	}
private:
	k_AdaptiveEntry* E;
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

class k_sPpmTracer : public k_Tracer<true, true>, public IRadiusProvider
{
private:
	e_SpatialLinkedMap<k_pPpmPhoton> m_sSurfaceMap;
	IVolumeEstimator* m_pVolumeEstimator;

	float m_fLightVisibility;

	float m_fInitialRadius;
	unsigned int m_uPhotonEmittedPass;
	unsigned long long m_uTotalPhotonsEmitted;

	unsigned int m_uBlocksPerLaunch;

	k_AdaptiveEntry* m_pEntries;
	float r_min, r_max;

	unsigned int k_Intial;
	float m_fIntitalRadMin, m_fIntitalRadMax;
public:
	bool m_bFinalGather;
	bool m_bDirect;

	k_sPpmTracer();
	virtual ~k_sPpmTracer()
	{
		m_sSurfaceMap.Free();
		CUDA_FREE(m_pEntries);
	}
	virtual void Resize(unsigned int _w, unsigned int _h);
	virtual void Debug(e_Image* I, const Vec2i& pixel);
	virtual void PrintStatus(std::vector<std::string>& a_Buf) const;
	virtual void CreateSliders(SliderCreateCallback a_Callback) const;
	virtual float getCurrentRadius(float exp) const
	{
		return CudaTracerLib::getCurrentRadius(m_fInitialRadius, m_uPassesDone, exp);
	}
	void getRadiusAt(int x, int y, float& r, float& rd) const;
	void getCurrentRMinRMax(float& rMin, float& rMax) const
	{
		rMin = CudaTracerLib::getCurrentRadius(r_min, m_uPassesDone, 2);
		rMax = CudaTracerLib::getCurrentRadius(r_max, m_uPassesDone, 2);
	}
protected:
	virtual void DoRender(e_Image* I);
	virtual void StartNewTrace(e_Image* I);
	virtual void RenderBlock(e_Image* I, int x, int y, int blockW, int blockH);
private:
	void doPhotonPass();
	void doPerPixelRadiusEstimation();
};

}