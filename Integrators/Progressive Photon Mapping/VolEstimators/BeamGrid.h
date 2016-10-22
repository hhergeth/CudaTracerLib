#pragma once
#include "PointStorage.h"
#include <functional>

namespace CudaTracerLib {

struct BeamGrid : public PointStorage
{
	typedef std::function<float()> kNN_clb_t;

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

	kNN_clb_t m_kNNClb;

	CUDA_FUNC_IN static constexpr int DIM()
	{
		return 2;
	}

	BeamGrid(unsigned int gridDim, unsigned int numPhotons, kNN_clb_t clb, int N = 20)
		: PointStorage(gridDim, numPhotons, m_sBeamGridStorage), m_kNNClb(clb), m_sBeamGridStorage(Vec3u(gridDim), gridDim * gridDim * gridDim * (1 + N))
	{
		if (!m_kNNClb)
			throw std::runtime_error("Please provide a kNN callback!");
	}

	virtual void Free()
	{
		PointStorage::Free();
		m_sBeamGridStorage.Free();
	}

	virtual void StartNewPass(DynamicScene* scene);

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

	template<bool USE_GLOBAL> CUDA_FUNC_IN Spectrum L_Volume(float rad, float NumEmitted, const NormalizedT<Ray>& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr, float& pl_est);
};

}