#include "StdAfx.h"
#include <Base/Buffer.h>
#include "SceneBVH.h"
#include "Mesh.h"
#include <SceneTypes/Node.h>
#include "SpatialStructures/SplitBVHBuilder.hpp"
#include "SpatialStructures/BVHRebuilder.h"

namespace CudaTracerLib {

bool SceneBVH::Build(Stream<Node>* nodStream, Buffer<Mesh, KernelMesh>* mesh_buf)
{
	class provider : public ISpatialInfoProvider
	{
		Stream<Node>* a_Nodes;
		Buffer<Mesh, KernelMesh>* mesh_buf;
		StreamReference<float4x4> a_Transforms;
	public:
		provider(Stream<Node>* A, Buffer<Mesh, KernelMesh>* B, StreamReference<float4x4> C)
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
	if (!nodStream->hasElements())
	{
		m_pBuilder->SetEmpty();
		return false;
	}
	provider p(nodStream, mesh_buf, tr_ref);
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

SceneBVH::SceneBVH(size_t a_NodeCount)
{
	m_pNodes = new Stream<BVHNodeData>(a_NodeCount * 2);//largest binary tree has the same amount of inner nodes
	m_pTransforms = new Stream<float4x4>(a_NodeCount);
	m_pInvTransforms = new Stream<float4x4>(a_NodeCount);
	tr_ref = m_pTransforms->malloc(m_pTransforms->getBufferLength());
	iv_tr_ref = m_pInvTransforms->malloc(m_pInvTransforms->getBufferLength());
	node_ref = m_pNodes->malloc(m_pNodes->getBufferLength());
	for (unsigned int i = 0; i < a_NodeCount; i++)
		*tr_ref(i) = *iv_tr_ref(i) = float4x4::Identity();
	m_pBuilder = new BVHRebuilder(node_ref(), node_ref.getLength(), (unsigned int)a_NodeCount, 0, 0);
}

SceneBVH::~SceneBVH()
{
	delete m_pNodes;
	delete m_pTransforms;
	delete m_pInvTransforms;
	delete m_pBuilder;
}

void SceneBVH::setTransform(BufferReference<Node, Node> n, const float4x4& mat)
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

KernelSceneBVH SceneBVH::getData(bool devicePointer)
{
	KernelSceneBVH q;
	q.m_uNumNodes = (unsigned int)m_pBuilder->getNumBVHNodesUsed();
	q.m_pNodes = m_pNodes->getKernelData(devicePointer).Data;
	q.m_sStartNode = m_pBuilder->getStartNode();
	q.m_pNodeTransforms = m_pTransforms->getKernelData(devicePointer).Data;
	q.m_pInvNodeTransforms = m_pInvTransforms->getKernelData(devicePointer).Data;
	return q;
}

size_t SceneBVH::getDeviceSizeInBytes()
{
	return m_pNodes->getDeviceSizeInBytes() + m_pTransforms->getDeviceSizeInBytes() + m_pInvTransforms->getDeviceSizeInBytes();
}

const float4x4& SceneBVH::getNodeTransform(BufferReference<Node, Node> n)
{
	return *m_pTransforms->operator()(n.getIndex());
}

BVHNodeData* SceneBVH::getBVHNode(unsigned int i)
{
	if (i >= m_pBuilder->getNumBVHNodesUsed())
		throw std::runtime_error(__FUNCTION__);
	StreamReference<BVHNodeData> n = m_pNodes->operator()(i);
	BVHNodeData* n2 = n.operator BVHNodeData *();
	return n2;
}

void SceneBVH::invalidateNode(BufferReference<Node, Node> n)
{
	m_pBuilder->invalidateNode(n.getIndex());
}

void SceneBVH::addNode(BufferReference<Node, Node> n)
{
	m_pBuilder->addNode(n.getIndex());
}

void SceneBVH::removeNode(BufferReference<Node, Node> n)
{
	m_pBuilder->removeNode(n.getIndex());
}

bool SceneBVH::needsBuild()
{
	return m_pBuilder->needsBuild();
}

AABB SceneBVH::getSceneBox()
{
	if (needsBuild())
		throw std::runtime_error(__FUNCTION__);
	return m_pBuilder->getBox();
}

void SceneBVH::printGraph(const std::string& path)
{
	m_pBuilder->printGraph(path);
}

}