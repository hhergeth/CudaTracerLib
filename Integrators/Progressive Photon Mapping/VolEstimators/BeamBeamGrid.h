#pragma once
#include "Beam.h"
#include <Engine/SpatialGrid.h>

namespace CudaTracerLib {

struct BeamBeamGrid : public IVolumeEstimator
{
	SpatialLinkedMap<int> m_sStorage;

	SynchronizedBuffer<Beam> m_sBeamStorage;

	unsigned int m_uBeamIdx;

	float m_fCurrentRadiusVol;

	BeamBeamGrid(unsigned int gridDim, unsigned int numBeams, int N = 100)
		: IVolumeEstimator(m_sStorage, m_sBeamStorage), m_sStorage(Vec3u(gridDim), gridDim * gridDim * gridDim * N), m_sBeamStorage(numBeams)
	{
		
	}

	virtual void Free()
	{
		m_sStorage.Free();
		m_sBeamStorage.Free();
	}

	CTL_EXPORT virtual void StartNewPass(const IRadiusProvider* radProvider, DynamicScene* scene);

	virtual void StartNewRendering(const AABB& box)
	{
		m_sStorage.SetSceneDimensions(box);
	}

	CUDA_FUNC_IN bool isFullK() const
	{
		return m_uBeamIdx >= m_sBeamStorage.getLength();
	}

	virtual bool isFull() const
	{
		return isFullK();
	}

	virtual void getStatusInfo(size_t& length, size_t& count) const
	{
		length = m_sBeamStorage.getLength();
		count = m_uBeamIdx;
	}

	virtual void PrintStatus(std::vector<std::string>& a_Buf) const
	{
		a_Buf.push_back(format("%.2f%% Beam grid indices", (float)m_sStorage.getNumStoredEntries() / m_sStorage.getNumEntries() * 100));
		a_Buf.push_back(format("%.2f%% Beams", (float)m_uBeamIdx / m_sBeamStorage.getLength() * 100));
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

	template<bool USE_GLOBAL> CUDA_FUNC_IN Spectrum L_Volume(float NumEmitted, const NormalizedT<Ray>& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, VolumeModel& model, PPM_Radius_Type radType, Spectrum& Tr);

	CUDA_FUNC_IN void Compute_kNN_radii(float numEmitted, float rad, float kToFind, const NormalizedT<Ray>& r, float tmin, float tmax, VolumeModel& model);
};

}