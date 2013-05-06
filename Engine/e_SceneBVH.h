#pragma once

#include "e_DataStream.h"
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
	e_DataStream<e_BVHNodeData>* m_pNodes;
	e_DataStream<float4x4>* m_pTransforms;
	e_DataStream<float4x4>* m_pInvTransforms;
	int startNode;
	AABB m_sBox;
public:
	e_SceneBVH(unsigned int a_NodeCount)
	{
		m_pNodes = new e_DataStream<e_BVHNodeData>(a_NodeCount * 2);//largest binary tree has the same amount of inner nodes
		m_pTransforms = new e_DataStream<float4x4>(a_NodeCount);
		m_pInvTransforms = new e_DataStream<float4x4>(a_NodeCount);
		startNode = -1;
		m_sBox = AABB::Identity();
	}
	~e_SceneBVH()
	{
		delete m_pNodes;
		delete m_pTransforms;
		delete m_pInvTransforms;
	}
	void Build(e_Node* a_Nodes, unsigned int a_Count);
	e_KernelSceneBVH getData()
	{
		e_KernelSceneBVH q;
		q.m_pNodes = m_pNodes->getDevice(0);
		q.m_sStartNode = startNode;
		q.m_pNodeTransforms = m_pTransforms->getDevice(0);
		q.m_pInvNodeTransforms = m_pInvTransforms->getDevice(0);
		return q;
	}
	unsigned int getSizeInBytes()
	{
		return m_pNodes->getSizeInBytes();
	}
private:

};
