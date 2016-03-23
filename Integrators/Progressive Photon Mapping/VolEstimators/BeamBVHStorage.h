#pragma once
#include "Beam.h"
#include <Engine/TriIntersectorData.h>
#include <Kernel/BVHTracer.h>
#include <vector>

namespace CudaTracerLib {

class DynamicScene;

class BeamBVHStorage : public IVolumeEstimator
{
	class BuilderCLB;
public:
	Beam* m_pDeviceBeams;
	Beam* m_pHostBeams;
	unsigned int m_uBeamIdx;
	unsigned int m_uNumBeams;

	BVHNodeData* m_pDeviceNodes;
	unsigned int m_uDeviceNumNodes;
	std::vector<BVHNodeData> m_sHostNodes;

	struct BeamRef
	{
		float t_min, t_max;
		Beam beam;
		CUDA_FUNC_IN BeamRef()
		{

		}
		CUDA_FUNC_IN BeamRef(const Beam& beam, float a, float b)
			: idx(0), t_min(a), t_max(b), beam(beam)
		{

		}
		CUDA_FUNC_IN void setLast()
		{
			idx = 1;
		}
		CUDA_FUNC_IN bool isLast() const
		{
			return idx != 0;
		}
		CUDA_FUNC_IN unsigned int getIdx() const
		{
			return UINT_MAX;
		}
	private:
		unsigned int idx;
	};
	std::vector<BeamRef> m_sHostRefs, m_sHostReorderedRefs;
	unsigned int m_uNumDeviceRefs;
	BeamRef* m_pDeviceRefs;

	DynamicScene* m_pScene;
	float m_fCurrentRadiusVol;
	AABB volBox;

	BeamBVHStorage(){}
	CTL_EXPORT BeamBVHStorage(unsigned int nBeams);
	CTL_EXPORT virtual void Free();

	virtual void StartNewPass(const IRadiusProvider* radProvider, DynamicScene* scene)
	{
		m_pScene = scene;
		m_uBeamIdx = 0;
		m_fCurrentRadiusVol = radProvider->getCurrentRadius(1);
	}

	virtual void StartNewRendering(const AABB& box)
	{
		volBox = box;
	}

	CUDA_FUNC_IN bool isFullK() const
	{
		return m_uBeamIdx >= m_uNumBeams;
	}

	virtual void getStatusInfo(size_t& length, size_t& count) const
	{
		length = m_uNumBeams;
		count = m_uBeamIdx;
	}

	virtual bool isFull() const
	{
		return isFullK();
	}

	virtual size_t getSize() const
	{
		return sizeof(*this);
	}

	virtual void PrintStatus(std::vector<std::string>& a_Buf) const
	{
		a_Buf.push_back(format("%.2f%% Vol Beams", (float)m_uBeamIdx / m_uNumBeams * 100));
	}

	CTL_EXPORT virtual void PrepareForRendering();

	CUDA_ONLY_FUNC bool StoreBeam(const Beam& b)
	{
		unsigned int i = atomicInc(&m_uBeamIdx, (unsigned int)-1);
		if (i < m_uNumBeams)
		{
			m_pDeviceBeams[i] = b;
			return true;
		}
		else return false;
	}

	CUDA_ONLY_FUNC bool StorePhoton(const Vec3f& pos, const Vec3f& wi, const Spectrum& phi)
	{
		return false;
	}

	template<typename CLB> CUDA_FUNC_IN void iterateBeams(const Ray& r, float rayEnd, const CLB& clb) const
	{
#ifdef ISCUDA
		BVHNodeData* host = 0;
#else
		const BVHNodeData* host = &m_sHostNodes[0];
#endif
		float rayT = rayEnd;
		TracerayTemplate(r, rayT, [&](int ref_idx)
		{
			do
			{
				clb(ref_idx);
			} while (!m_pDeviceRefs[ref_idx++].isLast());
			return false;
		}, host, m_pDeviceNodes);
	}

	template<bool USE_GLOBAL> CUDA_FUNC_IN Spectrum L_Volume(float NumEmitted, const NormalizedT<Ray>& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr) const;
};

}