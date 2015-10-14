#include <StdAfx.h>
#include "k_BeamBVHStorage.h"
#include "../../CudaMemoryManager.h"
#include "../../Engine/SceneBuilder/SplitBVHBuilder.hpp"
#include "../../Engine/e_DynamicScene.h"

k_BeamBVHStorage::k_BeamBVHStorage(unsigned int nBeams, e_DynamicScene* S)
	: m_uNumNodes(0), m_uBeamIdx(-1), m_uNumDeviceBVHBeams(0), m_pScene(S)
{
	m_uNumBeams = nBeams;
	CUDA_MALLOC(&m_pDeviceBeams, m_uNumBeams * sizeof(k_Beam));
	m_pHostBeams = new k_Beam[m_uNumBeams];
	m_sHostBVHBeams = new std::vector<k_Beam>();

}

void k_BeamBVHStorage::Free()
{
	CUDA_FREE(m_pDeviceBeams);
	delete[] m_pHostBeams;
	if (m_pDeviceNodes)
		CUDA_FREE(m_pDeviceNodes);
	if (m_pHostNodes)
		delete[] m_pHostNodes;
	if (m_pDeviceBVHBeams)
		CUDA_FREE(m_pDeviceBVHBeams);
	if (m_sHostBVHBeams)
		delete m_sHostBVHBeams;
}

void k_BeamBVHStorage::PrepareForRendering()
{
	class BuilderCLB : public IBVHBuilderCallback
	{
		k_Beam* m_pHostBeams;
		unsigned int N;
	public:
		std::vector<k_Beam> m_sReorderedBeams;
		std::vector<e_BVHNodeData> m_sBVHNodes;
		float r;
		BuilderCLB(k_Beam* B, unsigned int nBeams, float r)
			: m_pHostBeams(B), N(nBeams), r(r)
		{
			m_sReorderedBeams.reserve(nBeams);
			m_sBVHNodes.reserve(nBeams * 2);
		}
		virtual void getBox(unsigned int index, AABB* out) const
		{
			Vec3f a = m_pHostBeams[index].pos, b = a + m_pHostBeams[index].dir * m_pHostBeams[index].t;
			out->minV = out->maxV = a - Vec3f(r);
			*out = out->Extend(b + Vec3f(r));
		}
		virtual void iterateObjects(std::function<void(unsigned int)> f)
		{
			for (unsigned int i = 0; i < N; i++)
				f(i);
		}
		virtual unsigned int handleLeafObjects(unsigned int pNode)
		{
			m_sReorderedBeams.push_back(m_pHostBeams[pNode]);
			return (unsigned int)m_sReorderedBeams.size() - 1;
		}
		virtual void handleLastLeafObject(int parent)
		{
			m_sReorderedBeams.back().lastEntry = 1;
		}
		virtual e_BVHNodeData* HandleNodeAllocation(int* index)
		{
			m_sBVHNodes.push_back(e_BVHNodeData());
			*index = (unsigned int)(m_sBVHNodes.size() - 1) * 4;
			return &m_sBVHNodes.back();
		}
		virtual void HandleStartNode(int startNode)
		{

		}
		virtual void setSibling(int idx, int sibling)
		{

		}
	};

	m_uBeamIdx = min(m_uBeamIdx, m_uNumBeams);
	CUDA_MEMCPY_TO_HOST(m_pHostBeams, m_pDeviceBeams, m_uBeamIdx * sizeof(k_Beam));

	//shorten beams to create better bvh
	m_sHostBVHBeams->clear();
	m_sHostBVHBeams->reserve(size_t(m_uBeamIdx * m_pHostBeams->t / (2.0f * m_fCurrentRadiusVol)));
	auto data = m_pScene->getKernelSceneData(false);
	for (size_t i = 0; i < m_uBeamIdx; i++)
	{
		float t = 0;
		auto& b = m_pHostBeams[i];
		Spectrum tau(1.0f);
		while (t < b.t)
		{
			tau += data.m_sVolume.tau(Ray(b.pos, b.dir), t, t + 2 * m_fCurrentRadiusVol);
			m_sHostBVHBeams->push_back(k_Beam(b.pos + b.dir * t, b.dir, min(2 * m_fCurrentRadiusVol, b.t - t), b.Phi * (-tau).exp()));
			t += 2 * m_fCurrentRadiusVol;
		}
	}

	BuilderCLB clb(&m_sHostBVHBeams->operator[](0), (unsigned int)m_sHostBVHBeams->size(), m_fCurrentRadiusVol);
	auto plat = SplitBVHBuilder::Platform();
	SplitBVHBuilder builder(&clb, plat, SplitBVHBuilder::BuildParams());
	builder.run();
	if (clb.m_sReorderedBeams.size() > m_uNumDeviceBVHBeams)
	{
		if (m_uNumDeviceBVHBeams)
			CUDA_FREE(m_pDeviceBVHBeams);
		m_uNumDeviceBVHBeams = (unsigned int)clb.m_sReorderedBeams.size();
		CUDA_MALLOC(&m_pDeviceBVHBeams, m_uNumDeviceBVHBeams * sizeof(k_Beam));
	}
	CUDA_MEMCPY_TO_DEVICE(m_pDeviceBVHBeams, &clb.m_sReorderedBeams[0], (unsigned int)clb.m_sReorderedBeams.size() * sizeof(k_Beam));
	if (m_uNumNodes < clb.m_sBVHNodes.size())
	{
		if (m_uNumNodes)
		{
			CUDA_FREE(m_pDeviceNodes);
			delete[] m_pHostNodes;
		}
		m_uNumNodes = (unsigned int)clb.m_sBVHNodes.size();
		CUDA_MALLOC(&m_pDeviceNodes, m_uNumNodes * sizeof(e_BVHNodeData));
		m_pHostNodes = new e_BVHNodeData[m_uNumNodes];
	}
	memcpy(m_pHostNodes, &clb.m_sBVHNodes[0], m_uNumNodes);
	CUDA_MEMCPY_TO_DEVICE(m_pDeviceNodes, &clb.m_sBVHNodes[0], (unsigned int)clb.m_sBVHNodes.size() * sizeof(e_BVHNodeData));
}