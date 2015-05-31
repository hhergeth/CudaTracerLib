#include "StdAfx.h"
#include "e_SceneBVH.h"
#include "e_Mesh.h"
#include "e_Node.h"
#include "SceneBuilder/SplitBVHBuilder.hpp"

struct e_SceneBVH::BVHNodeInfo
{
	int parent;
};

class e_SceneBVH::BuilderCLB : public IBVHBuilderCallback
{
	e_StreamReference(e_Node) a_Nodes;
	e_BufferReference<e_Mesh, e_KernelMesh> a_Meshes;
	e_SceneBVH* t;
	int nodesAllocated;
public:
	BuilderCLB(e_StreamReference(e_Node) a, e_BufferReference<e_Mesh, e_KernelMesh> b, e_SceneBVH* c)
		: a_Nodes(a), a_Meshes(b), t(c), nodesAllocated(0)
	{
	}
	virtual unsigned int Count() const
	{
		return a_Nodes.getLength();
	}
	virtual void getBox(unsigned int index, AABB* out) const
	{
		unsigned int mi = a_Nodes(index)->m_uMeshIndex;
		AABB box = a_Meshes(mi)->m_sLocalBox;
		float4x4 mat = *t->m_pTransforms[0](index);
		*out = box.Transform(mat);
	}
	virtual void HandleBoundingBox(const AABB& box)
	{
		t->m_sBox = box;
	}
	virtual e_BVHNodeData* HandleNodeAllocation(int* index)
	{
		*index = nodesAllocated * 4;
		return t->m_pNodes->operator()(nodesAllocated++);
	}
	virtual unsigned int handleLeafObjects(unsigned int pNode)
	{
		return pNode;
	}
	virtual void handleLastLeafObject()
	{
	}
	virtual void HandleStartNode(int startNode)
	{
		t->startNode = startNode;
	}
};

bool e_SceneBVH::Build(e_StreamReference(e_Node) a_Nodes, e_BufferReference<e_Mesh, e_KernelMesh> a_Meshes)
{
	bool modified = false;
	if (a_Nodes.getLength() != 0 && needsBuild())
	{
		size_t m = max(nodesToInsert.size(), nodesToRecompute.size(), nodesToRemove.size());
		if (m > a_Nodes.getLength() / 2 || startNode == -1)
		{
			BuilderCLB b(a_Nodes, a_Meshes, this);
			SplitBVHBuilder::Platform Pq;
			Pq.m_maxLeafSize = 1;
			SplitBVHBuilder bu(&b, Pq, SplitBVHBuilder::BuildParams()); bu.run();
		}
		else
		{

		}
	}
	if (!a_Nodes.getLength())
		startNode = -1;
	m_pNodes->Invalidate();
	m_pNodes->UpdateInvalidated();
	m_pTransforms->Invalidate();
	m_pTransforms->UpdateInvalidated();
	m_pInvTransforms->Invalidate();
	m_pInvTransforms->UpdateInvalidated();
	nodesToRecompute.clear();
	nodesToInsert.clear();
	nodesToRemove.clear();
	return modified;
}

e_SceneBVH::e_SceneBVH(unsigned int a_NodeCount)
	: startNode(-1)
{
	m_pNodes = new e_Stream<e_BVHNodeData>(a_NodeCount * 2);//largest binary tree has the same amount of inner nodes
	m_pTransforms = new e_Stream<float4x4>(a_NodeCount);
	m_pInvTransforms = new e_Stream<float4x4>(a_NodeCount);
	startNode = -1;
	m_sBox = AABB::Identity();
	for(unsigned int i = 0; i < a_NodeCount; i++)
	{
		*m_pTransforms[0](i).operator->() = *m_pInvTransforms[0](i).operator->() = float4x4::Identity();
	}
	//INVALIDATE
	m_pTransforms->malloc(m_pTransforms->getLength());
	m_pInvTransforms->malloc(m_pInvTransforms->getLength());
	m_pNodes->malloc(m_pNodes->getLength());
	nodeToBVHNode.resize(a_NodeCount);
	bvhNodeData.resize(m_pNodes->getLength());
}

void e_SceneBVH::invalidateNode(e_BufferReference<e_Node, e_Node> n)
{
	nodesToRecompute.push_back(n);
}

void e_SceneBVH::setTransform(e_BufferReference<e_Node, e_Node> n, const float4x4& mat)
{
	unsigned int nodeIdx = n.getIndex();
	*m_pTransforms[0](nodeIdx).operator->() = mat;
	*m_pInvTransforms[0](nodeIdx).operator->() = mat.inverse();
	m_pTransforms->Invalidate(nodeIdx, 1);
	m_pInvTransforms->Invalidate(nodeIdx, 1);
	invalidateNode(n);
}

e_SceneBVH::~e_SceneBVH()
{
	delete m_pNodes;
	delete m_pTransforms;
	delete m_pInvTransforms;
}

e_KernelSceneBVH e_SceneBVH::getData(bool devicePointer)
{
	e_KernelSceneBVH q;
	q.m_pNodes = m_pNodes->getKernelData(devicePointer).Data;
	q.m_sStartNode = startNode;
	q.m_pNodeTransforms = m_pTransforms->getKernelData(devicePointer).Data;
	q.m_pInvNodeTransforms = m_pInvTransforms->getKernelData(devicePointer).Data;
	return q;
}

unsigned int e_SceneBVH::getSizeInBytes()
{
	return m_pNodes->getSizeInBytes();
}

const float4x4& e_SceneBVH::getNodeTransform(e_BufferReference<e_Node, e_Node> n)
{
	return *m_pTransforms->operator()(n.getIndex());
}

e_BVHNodeData& e_SceneBVH::getBVHNode(unsigned int i)
{
	return *m_pNodes->operator()(i).operator e_BVHNodeData *();
}

void e_SceneBVH::addNode(e_BufferReference<e_Node, e_Node> n)
{
	nodesToInsert.push_back(n);
}

void e_SceneBVH::removeNode(e_BufferReference<e_Node, e_Node> n)
{
	nodesToRemove.push_back(n);
}

bool e_SceneBVH::needsBuild()
{
	return nodesToRecompute.size() != 0 || nodesToInsert.size() != 0 || nodesToRemove.size() != 0;
}