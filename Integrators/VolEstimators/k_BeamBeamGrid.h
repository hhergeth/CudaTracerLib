#pragma once
#include "k_Beam.h"
#include <Engine/e_SpatialGrid.h>

struct k_BeamBeamGrid : public IVolumeEstimator
{
	e_SpatialLinkedMap<int> m_sStorage;

	k_Beam* m_pDeviceBeams;
	unsigned int m_uBeamIdx;
	unsigned int m_uBeamLength;

	unsigned int m_uNumEmitted;
	float m_fCurrentRadiusVol;

	k_BeamBeamGrid(unsigned int gridDim, unsigned int numBeams, int N = 100)
		: m_sStorage(gridDim, gridDim * gridDim * gridDim * N), m_uBeamLength(numBeams)
	{
		CUDA_MALLOC(&m_pDeviceBeams, sizeof(k_Beam) * m_uBeamLength);
	}

	virtual void Free()
	{
		m_sStorage.Free();
		CUDA_FREE(m_pDeviceBeams);
	}

	virtual void StartNewPass(const IRadiusProvider* radProvider, e_DynamicScene* scene);

	virtual void StartNewRendering(const AABB& box, float a_InitRadius)
	{
		m_sStorage.SetSceneDimensions(box, a_InitRadius);
	}

	CUDA_FUNC_IN bool isFullK() const
	{
		return m_uBeamIdx >= m_uBeamLength;
	}

	virtual bool isFull() const
	{
		return isFullK();
	}

	virtual unsigned int getNumEmitted() const
	{
		return m_uNumEmitted;
	}

	virtual void PrintStatus(std::vector<std::string>& a_Buf) const
	{
		a_Buf.push_back(format("%.2f%% Beam grid indices", (float)m_sStorage.deviceDataIdx / m_sStorage.numData * 100));
		a_Buf.push_back(format("%.2f%% Beams", (float)m_uBeamIdx / m_uBeamLength * 100));
	}

	virtual size_t getSize() const
	{
		return sizeof(*this);
	}

	virtual void PrepareForRendering();

	CUDA_ONLY_FUNC void StoreBeam(const k_Beam& b, bool firstStore);

	CUDA_ONLY_FUNC void StorePhoton(const Vec3f& pos, const Vec3f& wi, const Spectrum& phi, bool firstStore)
	{

	}

	template<bool USE_GLOBAL> CUDA_FUNC_IN Spectrum L_Volume(float a_r, CudaRNG& rng, const Ray& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr);
};