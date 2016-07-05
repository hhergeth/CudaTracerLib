#pragma once

#include <Kernel/Tracer.h>
#include "../PhotonMapHelper.h"
#include <CudaMemoryManager.h>
#include "VolEstimators/BeamBeamGrid.h"
#include "VolEstimators/BeamBVHStorage.h"
#include "VolEstimators/BeamGrid.h"
#include <map>
#include <boost/variant.hpp>

namespace CudaTracerLib {

struct k_AdaptiveEntry
{
	float Sum_psi, Sum_psi2;
	float Sum_DI, Sum_E_DI, Sum_E_DI2;
	float Sum_pl;
	float r_std;

	CUDA_FUNC_IN float compute_rd(int iteration, int J, int totalPhotons)
	{
		float VAR_Lapl = Sum_E_DI2 / iteration - math::sqr(Sum_E_DI / iteration);
		return 1.9635f * math::sqrt(VAR_Lapl) * math::pow((float)iteration, -1.0f / 8.0f);
	}

	CUDA_FUNC_IN float compute_r(int iteration, int J, int totalPhotons)
	{
		float VAR_Psi = Sum_psi2 / iteration - math::sqr(Sum_psi / iteration);
		float k_2 = 10.0f * PI / 168.0f, k_22 = k_2 * k_2;
		float E_pl = Sum_pl / totalPhotons;
		float nom = (2.0f * math::sqrt(VAR_Psi)), denom = (PI * J * E_pl * k_22 * math::sqr(Sum_DI / iteration) * iteration);
		if (nom == 0 || denom == 0) return getCurrentRadius(r_std, iteration, 2);
		return math::pow(nom / denom, 1.0f / 6.0f);
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

typedef SpatialLinkedMap<PPPMPhoton> SurfaceMapT;

#define PTDM(X) X(Constant) X(kNN) X(Adaptive)
ENUMIZE(PPM_Radius_Type, PTDM)
#undef PTDM

class PPPMTracer : public Tracer<true, true>, public IRadiusProvider
{
private:
	SurfaceMapT m_sSurfaceMap;
	SurfaceMapT* m_sSurfaceMapCaustic;
	IVolumeEstimator* m_pVolumeEstimator;

	float m_fLightVisibility;

	float m_fInitialRadiusSurf, m_fInitialRadiusVol;
	unsigned int m_uPhotonEmittedPassSurface, m_uPhotonEmittedPassVolume;
	unsigned long long m_uTotalPhotonsEmitted;

	unsigned int m_uBlocksPerLaunch;

	//adaptive data
	SynchronizedBuffer<k_AdaptiveEntry>* m_adpBuffer;
	float r_min, r_max;

	unsigned int k_Intial;
	//used when computing intial radius from density
	float m_fIntitalRadMin, m_fIntitalRadMax;
	bool m_useDirectLighting;
	float m_fProbSurface, m_fProbVolume;
	float m_debugScaleVal;
public:

	PARAMETER_KEY(bool, Direct)
	PARAMETER_KEY(bool, FinalGathering)
	PARAMETER_KEY(bool, AdaptiveAccProb)
	PARAMETER_KEY(PPM_Radius_Type, RadiiComputationType)
	PARAMETER_KEY(float, VolRadiusScale)

	CTL_EXPORT PPPMTracer();
	CTL_EXPORT virtual ~PPPMTracer();
	CTL_EXPORT virtual void Resize(unsigned int _w, unsigned int _h);
	CTL_EXPORT virtual void PrintStatus(std::vector<std::string>& a_Buf) const;
	virtual float getCurrentRadius(float exp) const
	{
		return getCurrentRadius(exp, false);
	}
	float getCurrentRadius(float exp, bool surf) const;
	void getCurrentRMinRMax(float& rMin, float& rMax) const
	{
		rMin = CudaTracerLib::getCurrentRadius(r_min, m_uPassesDone, 2);
		rMax = CudaTracerLib::getCurrentRadius(r_max, m_uPassesDone, 2);
	}
	float& getDebugScaleVal()
	{
		return m_debugScaleVal;
	}

	std::map<std::string, boost::variant<int, float>> getPixelInfo(int x, int y) const;
protected:
	CTL_EXPORT virtual void DoRender(Image* I);
	CTL_EXPORT virtual void StartNewTrace(Image* I);
	CTL_EXPORT virtual void RenderBlock(Image* I, int x, int y, int blockW, int blockH);
	CTL_EXPORT virtual void DebugInternal(Image* I, const Vec2i& pixel);
private:
	CTL_EXPORT void doPhotonPass();
	CTL_EXPORT void doPerPixelRadiusEstimation();
};

}