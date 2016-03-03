#pragma once
#include "PointStorage.h"

namespace CudaTracerLib {

struct BeamGrid : public PointStorage
{
	struct entry
	{
		int i;

		entry()
		{

		}

		CUDA_FUNC_IN entry(int i)
			: i(i)
		{

		}

		CUDA_FUNC_IN bool getFlag() const
		{
			return (i >> 31) == 1;
		}

		CUDA_FUNC_IN void setFlag()
		{
			i |= 1 << 31;
		}

		CUDA_FUNC_IN int getIndex() const
		{
			return i & 0x7fffffff;
		}
	};

	SpatialLinkedMap<entry> m_sBeamGridStorage;

	float photonDensNum;

	CUDA_FUNC_IN BeamGrid()
	{

	}

	BeamGrid(unsigned int gridDim, unsigned int numPhotons, int N = 20, float nnSearch = 1)
		: PointStorage(gridDim, numPhotons), photonDensNum(nnSearch), m_sBeamGridStorage(Vec3u(gridDim), gridDim * gridDim * gridDim * (1 + N))
	{

	}

	virtual void Free()
	{
		PointStorage::Free();
		m_sBeamGridStorage.Free();
	}

	virtual void StartNewPass(const IRadiusProvider* radProvider, DynamicScene* scene)
	{
		PointStorage::StartNewPass(radProvider, scene);
		m_fCurrentRadiusVol = radProvider->getCurrentRadius(2);
		m_sBeamGridStorage.ResetBuffer();
	}

	virtual void StartNewRendering(const AABB& box)
	{
		PointStorage::StartNewRendering(box);
		m_sBeamGridStorage.SetSceneDimensions(box);
	}

	virtual size_t getSize() const
	{
		return sizeof(*this);
	}

	CTL_EXPORT virtual void PrepareForRendering();

	virtual void PrintStatus(std::vector<std::string>& a_Buf) const
	{
		PointStorage::PrintStatus(a_Buf);
		a_Buf.push_back(format("%.2f%% Beam indices", (float)m_sBeamGridStorage.getNumStoredEntries() / m_sBeamGridStorage.getNumEntries() * 100));
	}

	template<bool USE_GLOBAL> CUDA_FUNC_IN Spectrum L_Volume(float NumEmitted, CudaRNG& rng, const NormalizedT<Ray>& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr);
};

}