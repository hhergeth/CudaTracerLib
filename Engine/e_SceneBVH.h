#pragma once

#include "e_Buffer.h"
#include <MathTypes.h>

class e_Node;
struct e_BVHNodeData;
class e_Mesh;
struct e_KernelMesh;

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
	e_SceneBVH(unsigned int a_NodeCount);
	~e_SceneBVH();
	void Build(e_StreamReference(e_Node), e_BufferReference<e_Mesh, e_KernelMesh> a_Meshes);
	e_KernelSceneBVH getData(bool devicePointer = true);
	unsigned int getSizeInBytes()
	{
		return m_pNodes->getSizeInBytes();
	}
private:

};
