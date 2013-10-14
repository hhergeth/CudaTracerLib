#include "StdAfx.h"
#include "e_SceneBVH.h"
#include "e_Node.h"

void e_SceneBVH::Build(e_StreamReference(e_Node) a_Nodes, e_BufferReference<e_Mesh, e_KernelMesh> a_Meshes)
{
	class clb : public IBVHBuilderCallback
	{
		e_StreamReference(e_Node) a_Nodes;
		e_BufferReference<e_Mesh, e_KernelMesh> a_Meshes;
		e_SceneBVH* t;
		int nodesAllocated;
	public:
		clb(e_StreamReference(e_Node) a, e_BufferReference<e_Mesh, e_KernelMesh> b, e_SceneBVH* c)
			: a_Nodes(a), a_Meshes(b), t(c), nodesAllocated(0)
		{
		}
		virtual unsigned int Count() const
		{
			return a_Nodes.getLength();
		}
		virtual void getBox(unsigned int index, AABB* out) const
		{
			*out = a_Nodes(index)->getWorldBox(a_Meshes(a_Nodes(index)->m_uMeshIndex));
		}
		virtual void HandleBoundingBox(const AABB& box)
		{
			t->m_sBox = box;
		}
		virtual e_BVHNodeData* HandleNodeAllocation(int* index)
		{
			*index = nodesAllocated;
			return t->m_pNodes->operator()(nodesAllocated++);
		}
		virtual void HandleStartNode(int startNode)
		{
			t->startNode = startNode;
		}
	};

	clb b(a_Nodes, a_Meshes, this);
	BVHBuilder::BuildBVH(&b, BVHBuilder::Platform(1));
	for(unsigned int i = 0; i < a_Nodes.getLength(); i++)
	{
		m_pTransforms->operator()(i) = a_Nodes[i].getWorldMatrix();
		m_pInvTransforms->operator()(i) = a_Nodes[i].getWorldMatrix().Inverse();
	}

	m_pNodes->Invalidate();
	m_pNodes->UpdateInvalidated();
	m_pTransforms->Invalidate();
	m_pTransforms->UpdateInvalidated();
	m_pInvTransforms->Invalidate();
	m_pInvTransforms->UpdateInvalidated();
}

e_SceneBVH::e_SceneBVH(unsigned int a_NodeCount)
{
	m_pNodes = new e_Stream<e_BVHNodeData>(a_NodeCount * 2);//largest binary tree has the same amount of inner nodes
	m_pTransforms = new e_Stream<float4x4>(a_NodeCount);
	m_pInvTransforms = new e_Stream<float4x4>(a_NodeCount);
	startNode = -1;
	m_sBox = AABB::Identity();
	tr0 = m_pTransforms->malloc(m_pTransforms->getLength());
	tr1 = m_pInvTransforms->malloc(m_pInvTransforms->getLength());
	nds = m_pNodes->malloc(m_pNodes->getLength());
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