#pragma once

#include <Kernel/Tracer.h>
#include "../PhotonMapHelper.h"
#include <CudaMemoryManager.h>
#include "VolEstimators/BeamBeamGrid.h"
#include "VolEstimators/BeamGrid.h"
#include <map>
#include <boost/variant.hpp>
#include <Engine/BlockLoclizedCudaBuffer.h>
#include "AdaptiveHelper.h"

namespace CudaTracerLib {

struct k_AdaptiveStruct
{
	float r_min;
	float r_max;
	int w;
	k_AdaptiveStruct(float rmin, float rmax, const BlockLoclizedCudaBuffer<APPM_PixelData> entBuf, int w, int m_uPassesDone)
		: w(w), E(entBuf)
	{
		r_min = rmin * math::pow(float(m_uPassesDone), -1.0f / 6.0f);
		r_max = rmax * math::pow(float(m_uPassesDone), -1.0f / 6.0f);
	}
	CUDA_FUNC_IN APPM_PixelData& operator()(int x, int y)
	{
		return E(x, y);
	}
private:
	BlockLoclizedCudaBuffer<APPM_PixelData> E;
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
	BlockLoclizedCudaBuffer<APPM_PixelData>* m_pAdpBuffer;
	float r_min, r_max;

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
	PARAMETER_KEY(float, kNN_Neighboor_Num)

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
	void getPhotonsEmittedLastPass(float& nSurface, float& nVolume)
	{
		nSurface = m_uPhotonEmittedPassSurface;
		nVolume = m_uPhotonEmittedPassVolume;
	}

	std::map<std::string, boost::variant<int, float>> getPixelInfo(int x, int y) const;
	APPM_PixelData& getPixelData(int x, int y) const
	{
		return m_pAdpBuffer->operator()(x, y);
	}
protected:
	CTL_EXPORT virtual void DoRender(Image* I);
	CTL_EXPORT virtual void StartNewTrace(Image* I);
	CTL_EXPORT virtual void RenderBlock(Image* I, int x, int y, int blockW, int blockH);
	CTL_EXPORT virtual void DebugInternal(Image* I, const Vec2i& pixel);
	virtual float getSplatScale();
private:
	CTL_EXPORT void doPhotonPass(Image* I);
	CTL_EXPORT void doPerPixelRadiusEstimation();
};

}