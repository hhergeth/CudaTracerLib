#pragma once
#include "k_Beam.h"
#include "../../Engine/e_TriIntersectorData.h"
#include "../../Kernel/k_BVHTracer.h"
#include <vector>

class e_DynamicScene;

class k_BeamBVHStorage : public IVolumeEstimator
{
	unsigned int m_uNumEmitted;
	k_Beam* m_pDeviceBeams;
	k_Beam* m_pHostBeams;
	unsigned int m_uNumBeams;
	e_BVHNodeData* m_pDeviceNodes;
	e_BVHNodeData* m_pHostNodes;
	unsigned int m_uNumNodes;
	unsigned int m_uBeamIdx;
	k_Beam* m_pDeviceBVHBeams;
	unsigned int m_uNumDeviceBVHBeams;
	std::vector<k_Beam>* m_sHostBVHBeams;
	e_DynamicScene* m_pScene;
	float m_fCurrentRadiusVol;
public:
	k_BeamBVHStorage(){}
	k_BeamBVHStorage(unsigned int nBeams, e_DynamicScene* S);
	virtual void Free();

	virtual void StartNewPass(const IRadiusProvider* radProvider, e_DynamicScene* scene)
	{
		m_pScene = scene;
		m_uBeamIdx = 0;
		m_uNumEmitted = 0;
		m_fCurrentRadiusVol = radProvider->getCurrentRadius(1);
	}

	virtual void StartNewRendering(const AABB& box, float a_InitRadius)
	{

	}

	CUDA_FUNC_IN bool isFullK() const
	{
		return m_uBeamIdx >= m_uNumBeams;
	}

	virtual bool isFull() const
	{
		return isFullK();
	}

	virtual unsigned int getNumEmitted() const
	{
		return m_uNumEmitted;
	}

	virtual size_t getSize() const
	{
		return sizeof(*this);
	}

	virtual void PrintStatus(std::vector<std::string>& a_Buf) const
	{
		a_Buf.push_back(format("%.2f%% Vol Beams", (float)m_uBeamIdx / m_uNumBeams * 100));
	}

	virtual void PrepareForRendering();

	CUDA_ONLY_FUNC void StoreBeam(const k_Beam& b, bool firstStore)
	{
		unsigned int i = atomicInc(&m_uBeamIdx, (unsigned int)-1);
		if (i < m_uNumBeams)
		{
			m_pDeviceBeams[i] = b;
			if (firstStore)
				atomicInc(&m_uNumEmitted, (unsigned int)-1);
		}
	}

	CUDA_ONLY_FUNC void StorePhoton(const Vec3f& pos, const Vec3f& wi, const Spectrum& phi, bool firstStore)
	{

	}

	template<typename CLB> CUDA_DEVICE CUDA_HOST void iterateBeams(const Ray& r, float rayEnd, const CLB& clb) const
	{
		float rayT = rayEnd;
		k_TraceRayTemplate(r, rayT, [&](int beamIdx)
		{
			do
			{
				clb(m_pDeviceBVHBeams[beamIdx]);
				beamIdx++;
			} while (!m_pDeviceBVHBeams[beamIdx].lastEntry);
			return false;
		}, m_pHostNodes, m_pDeviceNodes);
	}

	template<bool USE_GLOBAL> CUDA_FUNC_IN Spectrum L_Volume(float a_r, CudaRNG& rng, const Ray& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr) const;
};