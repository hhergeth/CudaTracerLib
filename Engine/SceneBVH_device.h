#pragma once

namespace CudaTracerLib {

struct float4x4;
struct BVHNodeData;

struct KernelSceneBVH
{
	int m_sStartNode;
	unsigned int m_uNumNodes;
	BVHNodeData* m_pNodes;
	float4x4* m_pNodeTransforms;
	float4x4* m_pInvNodeTransforms;
};

}