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

CUDA_DEVICE CUDA_HOST void ComputeMinMaxRadiusForScene(const AABB& box, float& rmin, float& rmax);

struct k_AdaptiveStruct
{
private:
	//stores the radii for dim 1,2,3 and the lapl reduction
	float r_min[4];
	float r_max[4];
	int w;
	BlockLoclizedCudaBuffer<APPM_PixelData> E;
public:
	k_AdaptiveStruct(const AABB& box, const BlockLoclizedCudaBuffer<APPM_PixelData> entBuf, int w, int m_uPassesDone)
		: w(w), E(entBuf)
	{
		float rmin, rmax;
		ComputeMinMaxRadiusForScene(box, rmin, rmax);
		for (int i = 5; i <= 8; i++)
		{
			auto N_i = math::pow(m_uPassesDone, -1.0f / i);
			r_min[i - 5] = rmin * N_i;
			r_max[i - 5] = rmax * N_i;
		}
	}
	CUDA_FUNC_IN APPM_PixelData& operator()(int x, int y)
	{
		return E(x, y);
	}
	template<int DIM> CUDA_FUNC_IN float getMinRad() const
	{
		return r_min[DIM - 1];
	}
	template<int DIM> CUDA_FUNC_IN float getMaxRad() const
	{
		return r_max[DIM - 1];
	}
	CUDA_FUNC_IN float getMinRadDeriv() const
	{
		return r_min[3];
	}
	CUDA_FUNC_IN float getMaxRadDeriv() const
	{
		return r_max[3];
	}

	template<int DIM> CUDA_FUNC_IN float clampRadius(float rad) const
	{
		return math::clamp(rad, getMinRad<2>(), getMaxRad<2>());
	}
	CUDA_FUNC_IN float clampRadiusDeriv(float rad) const
	{
		return math::clamp(rad, getMinRadDeriv(), getMaxRadDeriv());
	}
};

enum
{
	PPM_Photons_Per_Thread = 6,
	PPM_BlockX = 32,
	PPM_BlockY = 6,
	PPM_MaxRecursion = 12,

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

	//used when computing intial radius from density
	bool m_useDirectLighting;
	float m_fProbSurface, m_fProbVolume;
	float m_debugScaleVal;
public:

	PARAMETER_KEY(bool, Direct)
	PARAMETER_KEY(bool, FinalGathering)
	PARAMETER_KEY(bool, AdaptiveAccProb)
	PARAMETER_KEY(PPM_Radius_Type, RadiiComputationType)
	PARAMETER_KEY(float, VolRadiusScale)
	PARAMETER_KEY(float, kNN_Neighboor_Num_Surf)
	PARAMETER_KEY(float, kNN_Neighboor_Num_Vol)

	CTL_EXPORT PPPMTracer();
	CTL_EXPORT virtual ~PPPMTracer();
	CTL_EXPORT virtual void Resize(unsigned int _w, unsigned int _h);
	CTL_EXPORT virtual void PrintStatus(std::vector<std::string>& a_Buf) const;
	virtual float getCurrentRadius(float exp) const
	{
		return getCurrentRadius(exp, false);
	}
	float getCurrentRadius(float exp, bool surf) const;
	void getCurrentRMinRMax(float& rMin, float& rMax) const;
	float& getDebugScaleVal()
	{
		return m_debugScaleVal;
	}
	void getPhotonsEmittedLastPass(float& nSurface, float& nVolume)
	{
		nSurface = (float)m_uPhotonEmittedPassSurface;
		nVolume = (float)m_uPhotonEmittedPassVolume;
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
	CTL_EXPORT k_AdaptiveStruct getAdaptiveData();
};

}