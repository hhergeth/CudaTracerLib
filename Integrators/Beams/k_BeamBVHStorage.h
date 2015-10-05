#pragma once
#include "../../Defines.h"
#include "../../MathTypes.h"
#include "../../Engine/e_TriIntersectorData.h"
#ifdef __CUDACC__
#include "../../Kernel/k_BVHTracer.h"
#endif
#include <vector>

struct k_Beam
{
	Vec3f pos;
	Vec3f dir;
	float t;
	Spectrum Phi;
	unsigned int lastEntry;
	k_Beam(){}
	CUDA_FUNC_IN k_Beam(const Vec3f& p, const Vec3f& d, float t, const Spectrum& ph)
		: pos(p), dir(d), t(t), Phi(ph), lastEntry(0)
	{

	}
};

class e_DynamicScene;

class k_BeamBVHStorage
{
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
	void*	t_BVHNodes;
public:
	k_BeamBVHStorage(){}
	k_BeamBVHStorage(unsigned int nBeams);
	void StartRendering()
	{
		m_uBeamIdx = 0;
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
		float rayT;
		k_TraceRayTemplate(r, rayT, [&](int beamIdx)
		{
			do
			{
				clb(m_pDeviceBVHBeams[beamIdx]);
				beamIdx++;
			} while (!m_pDeviceBVHBeams[beamIdx].lastEntry);
			return false;
		}, t_BVHNodes, m_pHostNodes);
	}
};