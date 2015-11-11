#include "StdAfx.h"
#include "e_Buffer.h"
#include "e_SceneBVH.h"
#include "e_Mesh.h"
#include "e_Node.h"
#include "SceneBuilder/SplitBVHBuilder.hpp"
#include "e_BVHRebuilder.h"

namespace CudaTracerLib {

bool e_SceneBVH::Build(e_Stream<e_Node>* node_stream, e_Buffer<e_Mesh, e_KernelMesh>* mesh_buf)
{
	class provider : public ISpatialInfoProvider
	{
		e_Stream<e_Node>* a_Nodes;
		e_Buffer<e_Mesh, e_KernelMesh>* mesh_buf;
		e_StreamReference<float4x4> a_Transforms;
	public:
		provider(e_Stream<e_Node>* A, e_Buffer<e_Mesh, e_KernelMesh>* B, e_StreamReference<float4x4> C)
			: a_Nodes(A), mesh_buf(B), a_Transforms(C)
		{

		}
		virtual AABB getBox(unsigned int idx)
		{
			unsigned int mi = a_Nodes->operator()(idx)->m_uMeshIndex;
			AABB box = mesh_buf->operator()(mi)->m_sLocalBox;
			float4x4 mat = *a_Transforms(idx);
			return box.Transform(mat);
		}
		virtual void iterateObjects(std::function<void(unsigned int)> f)
		{
			for (auto it : *a_Nodes)
				f(it.getIndex());
		}
	};
	if (!node_stream->hasElements())
	{
		m_pBuilder->SetEmpty();
		return false;
	}
	provider p(node_stream, mesh_buf, tr_ref);
	bool modified = m_pBuilder->Build(&p);
	if (modified)
	{
		m_pNodes->Invalidate();
		m_pNodes->UpdateInvalidated();
		m_pTransforms->Invalidate();
		m_pTransforms->UpdateInvalidated();
		m_pInvTransforms->Invalidate();
		m_pInvTransforms->UpdateInvalidated();
	}
	return modified;
}

e_SceneBVH::e_SceneBVH(size_t a_NodeCount)
{
	m_pNodes = new e_Stream<e_BVHNodeData>(a_NodeCount * 2);//largest binary tree has the same amount of inner nodes
	m_pTransforms = new e_Stream<float4x4>(a_NodeCount);
	m_pInvTransforms = new e_Stream<float4x4>(a_NodeCount);
	tr_ref = m_pTransforms->malloc(m_pTransforms->getBufferLength());
	iv_tr_ref = m_pInvTransforms->malloc(m_pInvTransforms->getBufferLength());
	node_ref = m_pNodes->malloc(m_pNodes->getBufferLength());
	for (unsigned int i = 0; i < a_NodeCount; i++)
		*tr_ref(i) = *iv_tr_ref(i) = float4x4::Identity();
	m_pBuilder = new e_BVHRebuilder(node_ref(), node_ref.getLength(), (unsigned int)a_NodeCount, 0, 0);
}

e_SceneBVH::~e_SceneBVH()
{
	delete m_pNodes;
	delete m_pTransforms;
	delete m_pInvTransforms;
}

void e_SceneBVH::setTransform(e_BufferReference<e_Node, e_Node> n, const float4x4& mat)
{
	if (n.getIndex() >= m_pTransforms->getBufferLength())
	{
		throw std::runtime_error("The number of nodes can not be enlarged!");
	}
	unsigned int nodeIdx = n.getIndex();
	*m_pTransforms[0](nodeIdx).operator->() = mat;
	*m_pInvTransforms[0](nodeIdx).operator->() = mat.inverse();
	m_pTransforms->Invalidate(nodeIdx, 1);
	m_pInvTransforms->Invalidate(nodeIdx, 1);
	invalidateNode(n);
}

e_KernelSceneBVH e_SceneBVH::getData(bool devicePointer)
{
	e_KernelSceneBVH q;
	q.m_uNumNodes = (unsigned int)m_pBuilder->getNumBVHNodesUsed();
	q.m_pNodes = m_pNodes->getKernelData(devicePointer).Data;
	q.m_sStartNode = m_pBuilder->getStartNode();
	q.m_pNodeTransforms = m_pTransforms->getKernelData(devicePointer).Data;
	q.m_pInvNodeTransforms = m_pInvTransforms->getKernelData(devicePointer).Data;
	return q;
}

size_t e_SceneBVH::getDeviceSizeInBytes()
{
	return m_pNodes->getDeviceSizeInBytes() + m_pTransforms->getDeviceSizeInBytes() + m_pInvTransforms->getDeviceSizeInBytes();
}

const float4x4& e_SceneBVH::getNodeTransform(e_BufferReference<e_Node, e_Node> n)
{
	return *m_pTransforms->operator()(n.getIndex());
}

e_BVHNodeData* e_SceneBVH::getBVHNode(unsigned int i)
{
	if (i >= m_pBuilder->getNumBVHNodesUsed())
		throw std::runtime_error(__FUNCTION__);
	e_StreamReference<e_BVHNodeData> n = m_pNodes->operator()(i);
	e_BVHNodeData* n2 = n.operator e_BVHNodeData *();
	return n2;
}

void e_SceneBVH::invalidateNode(e_BufferReference<e_Node, e_Node> n)
{
	m_pBuilder->invalidateNode(n.getIndex());
}

void e_SceneBVH::addNode(e_BufferReference<e_Node, e_Node> n)
{
	m_pBuilder->addNode(n.getIndex());
}

void e_SceneBVH::removeNode(e_BufferReference<e_Node, e_Node> n)
{
	m_pBuilder->removeNode(n.getIndex());
}

bool e_SceneBVH::needsBuild()
{
	return m_pBuilder->needsBuild();
}

AABB e_SceneBVH::getSceneBox()
{
	if (needsBuild())
		throw std::runtime_error(__FUNCTION__);
	return m_pBuilder->getBox();
}

void e_SceneBVH::printGraph(const std::string& path)
{
	m_pBuilder->printGraph(path);
}

}