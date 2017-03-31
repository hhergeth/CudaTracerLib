#pragma once

#include <Kernel/Tracer.h>
#include "../PhotonMapHelper.h"
#include <CudaMemoryManager.h>
#include "VolEstimators/BeamBeamGrid.h"
#include "VolEstimators/BeamGrid.h"
#include <Engine/BlockLoclizedCudaBuffer.h>
#include "SurfEstimators/EntryEstimator.h"

namespace CudaTracerLib {

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

typedef EntryEstimator SurfaceMapT;
typedef unsigned long long counter_t;

struct APPM_PixelData
{
	//recursive density estimator
	struct AdaptiveDensityEstimator
	{
		float density;//normalized density <=> Sum[k(t)]

		CUDA_FUNC_IN AdaptiveDensityEstimator()
			: density(0)
		{

		}

		CUDA_FUNC_IN void addSample(float k)
		{
			density += k;
		}

		CUDA_FUNC_IN float computeDensityEstimate(counter_t photonsEmitted, unsigned int N) const
		{
			return density / N;
		}
	};

	AdaptiveDensityEstimator surf_density;
	AdaptiveDensityEstimator vol_density;

	CUDA_FUNC_IN APPM_PixelData()
	{

	}
};

struct k_AdaptiveStruct
{
	PPM_Radius_Type m_radTypeSurf, m_radTypeVol;

	float m_radSurf, m_radVol[3];

	float m_surfMin, m_surfMax;
	float r_volMin[3], r_volMax[3];

	float kToFindSurf, kToFindVol;

	counter_t numPhotonsSurf, numPhotonsVol;

	unsigned int w, numIter;
	SynchronizedBuffer<APPM_PixelData> E;
public:
	k_AdaptiveStruct(float rSurfInitial, float rVolInitial, float surf_min, float surf_max, float vol_min, float vol_max, const SynchronizedBuffer<APPM_PixelData>& entBuf, unsigned int w, unsigned int m_uPassesDone, counter_t nSurf, counter_t nVol, float tarSurf, float tarVol, PPM_Radius_Type surfType, PPM_Radius_Type volType)
		: w(w), E(entBuf), kToFindSurf(tarSurf), kToFindVol(tarVol), numPhotonsSurf(nSurf), numPhotonsVol(nVol), numIter(m_uPassesDone), m_radTypeSurf(surfType), m_radTypeVol(volType), m_surfMin(surf_min), m_surfMax(surf_max)
	{
		m_radSurf = getCurrentRadius(rSurfInitial, numIter, 2);

		for (int i = 5; i <= 7; i++)
		{
			m_radVol[i - 5] = getCurrentRadius(rSurfInitial, numIter, i - 4.0f);

			auto N_i = math::pow((float)m_uPassesDone, -1.0f / i);
			r_volMin[i - 5] = vol_min * N_i;
			r_volMax[i - 5] = vol_max * N_i;
		}
	}

	CUDA_FUNC_IN APPM_PixelData& operator()(int x, int y)
	{
		return E[y * w + x];
	}

	CUDA_FUNC_IN float getMinRadSurf() const
	{
		return m_surfMin;
	}
	CUDA_FUNC_IN float getMaxRadSurf() const
	{
		return m_surfMax;
	}

	template<int DIM> CUDA_FUNC_IN float getMinRadVol() const
	{
		return r_volMin[DIM - 1];
	}
	template<int DIM> CUDA_FUNC_IN float getMaxRadVol() const
	{
		return r_volMax[DIM - 1];
	}

	template<int DIM> CUDA_FUNC_IN float clampRadiusVol(float rad) const
	{
		return math::clamp(rad, getMinRadVol<DIM>(), getMaxRadVol<DIM>());
	}

	float clampRadiusSurf(float rad) const
	{
		return math::clamp(rad, getMinRadSurf(), getMaxRadSurf());
	}

	CUDA_FUNC_IN float computekNNRadiusSurf(const APPM_PixelData& data) const
	{
		float density = data.surf_density.computeDensityEstimate(numPhotonsSurf, numIter);
		return density_to_rad<2>(kToFindSurf, density, m_surfMin, m_surfMax, numIter);
	}

	template<int DIM> CUDA_FUNC_IN float computekNNRadiusVol(const APPM_PixelData& data) const
	{
		float density = data.vol_density.computeDensityEstimate(numPhotonsVol, numIter);
		return density_to_rad<DIM>(kToFindVol, density, getMinRadVol<DIM>(), getMaxRadVol<DIM>(), numIter);
	}

	CUDA_FUNC_IN float getRadiusSurf(const APPM_PixelData& data) const
	{
		return m_radTypeSurf == PPM_Radius_Type::Constant || numIter <= 1 ? m_radSurf : computekNNRadiusSurf(data);
	}

	template<int DIM>  CUDA_FUNC_IN float getRadiusVol(const APPM_PixelData& data) const
	{
		return m_radTypeVol == PPM_Radius_Type::Constant || numIter <= 1 ? m_radVol[DIM - 1] : computekNNRadiusVol<DIM>(data);
	}
};

class PPPMTracer : public Tracer<true>
{
private:
	SurfaceMapT m_sSurfaceMap;
	SurfaceMapT* m_sSurfaceMapCaustic;
	IVolumeEstimator* m_pVolumeEstimator;

	float m_fLightVisibility;

	float m_fInitialRadiusSurf, m_fInitialRadiusVol;
	AABB m_boxSurf, m_boxVol;
	unsigned int m_uPhotonEmittedPassSurface, m_uPhotonEmittedPassVolume;
	counter_t m_uTotalPhotonsEmittedSurface, m_uTotalPhotonsEmittedVolume;

	unsigned int m_uBlocksPerLaunch;

	//per pixel data
	SynchronizedBuffer<APPM_PixelData>* m_pPixelBuffer;

	bool m_useDirectLighting;
	float m_fProbSurface, m_fProbVolume;
public:

	PARAMETER_KEY(bool, Direct)
	PARAMETER_KEY(int, N_FG_Samples)
	PARAMETER_KEY(bool, AdaptiveAccProb)
	PARAMETER_KEY(PPM_Radius_Type, RadiiComputationTypeSurf)
	PARAMETER_KEY(PPM_Radius_Type, RadiiComputationTypeVol)
	PARAMETER_KEY(float, VolRadiusScale)
	PARAMETER_KEY(float, kNN_Neighboor_Num_Surf)
	PARAMETER_KEY(float, kNN_Neighboor_Num_Vol)

	CTL_EXPORT PPPMTracer();
	CTL_EXPORT virtual ~PPPMTracer();
	CTL_EXPORT virtual void Resize(unsigned int _w, unsigned int _h);
	CTL_EXPORT virtual void PrintStatus(std::vector<std::string>& a_Buf) const;
	void getStartRadii(float& radSurf, float& radVol) const
	{
		radSurf = m_fInitialRadiusSurf;
		radVol = m_fInitialRadiusVol;
	}

	void getPhotonsEmittedLastPass(counter_t& nSurface, counter_t& nVolume)
	{
		nSurface = m_uPhotonEmittedPassSurface;
		nVolume = m_uPhotonEmittedPassVolume;
	}

	APPM_PixelData& getPixelData(int x, int y) const
	{
		m_pPixelBuffer->Synchronize();
		return m_pPixelBuffer->operator[](y * w + x);
	}
	CTL_EXPORT k_AdaptiveStruct getAdaptiveData() const;
	virtual float getSplatScale() const;
protected:
	CTL_EXPORT virtual void DoRender(Image* I);
	CTL_EXPORT virtual void StartNewTrace(Image* I);
	CTL_EXPORT virtual void RenderBlock(Image* I, int x, int y, int blockW, int blockH);
private:
	CTL_EXPORT void doPhotonPass(Image* I);
};

}