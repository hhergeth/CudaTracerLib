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
		virtual void HandleStartNode(int startNode)
		{
			t->startNode = startNode;
		}
	};

	clb b(a_Nodes, a_Meshes, this);
	BVHBuilder::Platform Pq;
	Pq.m_maxLeafSize = 1;
	BVHBuilder::BuildBVH(&b, Pq);

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
	for(unsigned int i = 0; i < a_NodeCount; i++)
	{
		*m_pTransforms[0](i).operator->() = *m_pInvTransforms[0](i).operator->() = float4x4::Identity();
	}
	//INVALIDATE
	m_pTransforms->malloc(m_pTransforms->getLength());
	m_pInvTransforms->malloc(m_pInvTransforms->getLength());
	m_pNodes->malloc(m_pNodes->getLength());
}


void e_SceneBVH::UpdateInvalidated()
{
	m_pNodes->UpdateInvalidated();
	m_pTransforms->UpdateInvalidated();
	m_pInvTransforms->UpdateInvalidated();
}

void e_SceneBVH::setTransform(unsigned int nodeIdx, const float4x4& mat)
{
	*m_pTransforms[0](nodeIdx).operator->() = mat;
	*m_pInvTransforms[0](nodeIdx).operator->() = mat.Inverse();
	m_pTransforms->Invalidate(nodeIdx, 1);
	m_pInvTransforms->Invalidate(nodeIdx, 1);
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