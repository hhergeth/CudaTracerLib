#pragma once

#include <Kernel/Tracer.h>
#include "../PhotonMapHelper.h"
#include <CudaMemoryManager.h>
#include "VolEstimators/BeamBeamGrid.h"
#include "VolEstimators/BeamBVHStorage.h"
#include "VolEstimators/BeamGrid.h"

namespace CudaTracerLib {

struct k_AdaptiveEntry
{
	float E_psi, E_psi2;
	float DI, E_DI, E_DI2;
	float pl;
	float r_std;

	CUDA_FUNC_IN float compute_rd(int iteration)
	{
		float VAR_Lapl = E_DI2 - E_DI * E_DI;
		return 1.9635f * math::sqrt(VAR_Lapl) * math::pow(iteration, -1.0f / 8.0f);
	}

	CUDA_FUNC_IN float compute_r(int iteration, int J, int totalPhotons)
	{
		float VAR_Psi = E_psi2 - E_psi * E_psi;
		float k_2 = 10.0f * PI / 168.0f, k_22 = k_2 * k_2;
		float E_pl = pl / totalPhotons;
		float ta = (2.0f * math::sqrt(VAR_Psi)) / (PI * E_pl * k_22 * E_DI * E_DI) / J;
		return math::pow(ta, 1.0f / 6.0f) * math::pow(iteration, -1.0f / 6.0f);
	}
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

class PPPMTracer : public Tracer<true, true>, public IRadiusProvider
{
private:
	SpatialLinkedMap<PPPMPhoton> m_sSurfaceMap;
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
	bool m_useDirectLighting;
public:

	PARAMETER_KEY(bool, Direct)
	PARAMETER_KEY(bool, PerPixelRadius)

	PPPMTracer();
	virtual ~PPPMTracer()
	{
		m_sSurfaceMap.Free();
		CUDA_FREE(m_pEntries);
	}
	virtual void Resize(unsigned int _w, unsigned int _h);
	virtual void Debug(Image* I, const Vec2i& pixel);
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
	virtual void DoRender(Image* I);
	virtual void StartNewTrace(Image* I);
	virtual void RenderBlock(Image* I, int x, int y, int blockW, int blockH);
private:
	void doPhotonPass();
	void doPerPixelRadiusEstimation();
};

}