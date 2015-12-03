#include <StdAfx.h>
#include "BeamBVHStorage.h"
#include <CudaMemoryManager.h>
#include <Engine/SceneBuilder/SplitBVHBuilder.hpp>
#include <Engine/DynamicScene.h>

namespace CudaTracerLib {

BeamBVHStorage::BeamBVHStorage(unsigned int nBeams)
	: m_uNumBeams(nBeams), m_uBeamIdx(0), m_uDeviceNumNodes(0), m_uNumDeviceRefs(0)
{
	CUDA_MALLOC(&m_pDeviceBeams, m_uNumBeams * sizeof(Beam));
	m_pHostBeams = new Beam[m_uNumBeams];
}

void BeamBVHStorage::Free()
{
	CUDA_FREE(m_pDeviceBeams);
	delete[] m_pHostBeams;
	if (m_pDeviceNodes)
		CUDA_FREE(m_pDeviceNodes);
	if (m_pDeviceRefs)
		delete[] m_pDeviceRefs;
}

class BeamBVHStorage::BuilderCLB : public IBVHBuilderCallback
{
	BeamBVHStorage* storage;
public:
	BuilderCLB(BeamBVHStorage* s)
		: storage(s)
	{

	}

	virtual void startConstruction(unsigned int nInnerNodes, unsigned int nLeafNodes)
	{
		storage->m_sHostNodes.clear();
		storage->m_sHostNodes.reserve(nInnerNodes);
		storage->m_sHostReorderedRefs.clear();
		storage->m_sHostReorderedRefs.reserve(nLeafNodes);
	}

	virtual void iterateObjects(std::function<void(unsigned int, const AABB&)> f)
	{
		for (unsigned int i = 0; i < storage->m_sHostRefs.size(); i++)
		{
			auto& r = storage->m_sHostRefs[i];
			auto& b = r.beam;//storage->m_pHostBeams[r.getIdx()]
			AABB box = b.getSegmentAABB(r.t_min, r.t_max, storage->m_fCurrentRadiusVol);
			f(i, box);
		}
	}

	virtual unsigned int createLeafNode(unsigned int parentBVHNodeIdx, const std::vector<unsigned int>& objIndices)
	{
		if (storage->m_sHostReorderedRefs.size() + objIndices.size() > storage->m_sHostReorderedRefs.capacity())
			throw std::runtime_error("Trying to add too many object indices!");
		unsigned int firstIdx = (unsigned int)storage->m_sHostReorderedRefs.size();
		for (size_t i = 0; i < objIndices.size(); i++)
		{
			storage->m_sHostReorderedRefs.push_back(storage->m_sHostRefs[objIndices[i]]);
		}
		storage->m_sHostReorderedRefs.back().setLast();
		return firstIdx;
	}

	virtual void finishConstruction(unsigned int startNode, const AABB& sceneBox)
	{

	}

	virtual unsigned int createInnerNode(BVHNodeData*& innerNode)
	{
		if (storage->m_sHostNodes.size() == storage->m_sHostNodes.capacity())
			throw std::runtime_error("Trying to add to many inner nodes!");
		storage->m_sHostNodes.push_back(BVHNodeData());
		innerNode = &storage->m_sHostNodes.back();
		return (unsigned int)(storage->m_sHostNodes.size() - 1);
	}
};

void BeamBVHStorage::PrepareForRendering()
{
	m_uBeamIdx = min(m_uBeamIdx, m_uNumBeams);
	CUDA_MEMCPY_TO_HOST(m_pHostBeams, m_pDeviceBeams, m_uBeamIdx * sizeof(Beam));

	//shorten beams to create better bvh
	auto data = m_pScene->getKernelSceneData(false);
	const int gridSize = 10;
	Vec3f invTargetSize = Vec3f(1.0f) / (volBox.Size() / gridSize);
	m_sHostRefs.clear();
	for (size_t i = 0; i < m_uBeamIdx; i++)
	{
		auto& b = m_pHostBeams[i];
		const AABB objaabb = b.getAABB(m_fCurrentRadiusVol);
		const int maxAxis = b.dir.abs().arg_max();
		const int chopCount = (int)(objaabb.Size()[maxAxis] * invTargetSize[maxAxis]) + 1;
		const float invChopCount = 1.0f / (float)chopCount;
		for (int chop = 0; chop < chopCount; ++chop)
		{
			float t_min = (chop)* invChopCount * b.t, t_max = (chop + 1) * invChopCount * b.t;
			m_sHostRefs.push_back(BeamRef(b, t_min, t_max));
		}
	}

	BuilderCLB clb(this);
	auto plat = SplitBVHBuilder::Platform();
	plat.m_spatialSplits = false;
	SplitBVHBuilder builder(&clb, plat, SplitBVHBuilder::BuildParams());
	builder.run();
	if (m_sHostReorderedRefs.size() > m_uNumDeviceRefs)
	{
		if (m_uNumDeviceRefs)
			CUDA_FREE(m_pDeviceRefs);
		m_uNumDeviceRefs = (unsigned int)m_sHostReorderedRefs.size();
		CUDA_MALLOC(&m_pDeviceRefs, m_uNumDeviceRefs * sizeof(BeamRef));
	}
	CUDA_MEMCPY_TO_DEVICE(m_pDeviceRefs, &m_sHostReorderedRefs[0], (unsigned int)m_sHostReorderedRefs.size() * sizeof(BeamRef));
	if (m_uDeviceNumNodes < m_sHostNodes.size())
	{
		if (m_uDeviceNumNodes)
			CUDA_FREE(m_pDeviceNodes);
		m_uDeviceNumNodes = (unsigned int)m_sHostNodes.size();
		CUDA_MALLOC(&m_pDeviceNodes, m_uDeviceNumNodes * sizeof(BVHNodeData));
	}
	CUDA_MEMCPY_TO_DEVICE(m_pDeviceNodes, &m_sHostNodes[0], (unsigned int)m_sHostNodes.size() * sizeof(BVHNodeData));
}

}