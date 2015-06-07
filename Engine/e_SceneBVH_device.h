#pragma once

struct float4x4;
struct e_BVHNodeData;

struct e_KernelSceneBVH
{
	int m_sStartNode;
	unsigned int m_uNumNodes;
	e_BVHNodeData* m_pNodes;
	float4x4* m_pNodeTransforms;
	float4x4* m_pInvNodeTransforms;
};