#pragma once
#include "Beam.h"
#include <Engine/SpatialGrid.h>

namespace CudaTracerLib {

struct BeamBeamGrid : public IVolumeEstimator
{
	SpatialLinkedMap<int> m_sStorage;

	Beam* m_pDeviceBeams;
	unsigned int m_uBeamIdx;
	unsigned int m_uBeamLength;

	float m_fCurrentRadiusVol;

	BeamBeamGrid(unsigned int gridDim, unsigned int numBeams, int N = 100)
		: m_sStorage(Vec3u(gridDim), gridDim * gridDim * gridDim * N), m_uBeamLength(numBeams)
	{
		CUDA_MALLOC(&m_pDeviceBeams, sizeof(Beam) * m_uBeamLength);
	}

	virtual void Free()
	{
		m_sStorage.Free();
		CUDA_FREE(m_pDeviceBeams);
	}

	CTL_EXPORT virtual void StartNewPass(const IRadiusProvider* radProvider, DynamicScene* scene);

	virtual void StartNewRendering(const AABB& box)
	{
		m_sStorage.SetSceneDimensions(box);
	}

	CUDA_FUNC_IN bool isFullK() const
	{
		return m_uBeamIdx >= m_uBeamLength;
	}

	virtual bool isFull() const
	{
		return isFullK();
	}

	virtual void getStatusInfo(size_t& length, size_t& count) const
	{
		length = m_uBeamLength;
		count = m_uBeamIdx;
	}

	virtual void PrintStatus(std::vector<std::string>& a_Buf) const
	{
		a_Buf.push_back(format("%.2f%% Beam grid indices", (float)m_sStorage.getNumStoredEntries() / m_sStorage.getNumEntries() * 100));
		a_Buf.push_back(format("%.2f%% Beams", (float)m_uBeamIdx / m_uBeamLength * 100));
	}

	virtual size_t getSize() const
	{
		return sizeof(*this);
	}

	CTL_EXPORT virtual void PrepareForRendering();

	CUDA_ONLY_FUNC bool StoreBeam(const Beam& b);

	CUDA_ONLY_FUNC bool StorePhoton(const Vec3f& pos, const Vec3f& wi, const Spectrum& phi)
	{
		return false;
	}

	template<bool USE_GLOBAL> CUDA_FUNC_IN Spectrum L_Volume(float NumEmitted, float radius, CudaRNG& rng, const NormalizedT<Ray>& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr);
};

}