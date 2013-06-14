#pragma once

#include "e_Buffer.h"
#include "..\Math\vector.h"
#include "e_Node.h"

struct e_KernelSceneBVH
{
	int m_sStartNode;
	e_BVHNodeData* m_pNodes;
	float4x4* m_pNodeTransforms;
	float4x4* m_pInvNodeTransforms;
};


class e_SceneBVH
{
public:
	e_Stream<e_BVHNodeData>* m_pNodes;
	e_Stream<float4x4>* m_pTransforms;
	e_Stream<float4x4>* m_pInvTransforms;
	int startNode;
	AABB m_sBox;

	e_StreamReference(float4x4) tr0, tr1;
	e_StreamReference(e_BVHNodeData) nds;
public:
	e_SceneBVH(unsigned int a_NodeCount)
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
	~e_SceneBVH()
	{
		delete m_pNodes;
		delete m_pTransforms;
		delete m_pInvTransforms;
	}
	void Build(e_StreamReference(e_Node), e_BufferReference<e_Mesh, e_KernelMesh> a_Meshes);
	e_KernelSceneBVH getData()
	{
		e_KernelSceneBVH q;
		q.m_pNodes = m_pNodes->getKernelData().Data;
		q.m_sStartNode = startNode;
		q.m_pNodeTransforms = m_pTransforms->getKernelData().Data;
		q.m_pInvNodeTransforms = m_pInvTransforms->getKernelData().Data;
		return q;
	}
	unsigned int getSizeInBytes()
	{
		return m_pNodes->getSizeInBytes();
	}
private:

};
