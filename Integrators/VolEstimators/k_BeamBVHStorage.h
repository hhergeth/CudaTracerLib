#pragma once
#include "k_Beam.h"
#include "../../Engine/e_TriIntersectorData.h"
#include "../../Kernel/k_BVHTracer.h"
#include <vector>

class e_DynamicScene;

class k_BeamBVHStorage
{
public:
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

	k_BeamBVHStorage(){}
	k_BeamBVHStorage(unsigned int nBeams);
	void StartRendering()
	{
		m_uBeamIdx = 0;
		m_uNumEmitted = 0;
	}
	CUDA_FUNC_IN float getNumEmitted() const
	{
		return m_uNumEmitted;
	}
	CUDA_ONLY_FUNC bool insertBeam(const k_Beam& b)
	{
#ifdef ISCUDA
		unsigned int i = atomicInc(&m_uBeamIdx, (unsigned int)-1);
		if(i < m_uNumBeams)
		{
			m_pDeviceBeams[i] = b;
			return true;
		}
		else return false;
#endif
	}
	void BuildStorage(float max_query_radius, e_DynamicScene* a_Scene);
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
};