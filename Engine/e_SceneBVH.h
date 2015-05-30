#pragma once

#include <MathTypes.h>
#include "e_SceneBVH_device.h"

class e_Node;
class e_Mesh;
struct e_KernelMesh;
template<typename T> class e_Stream;
template<typename H, typename D> class e_BufferReference;

class e_SceneBVH
{
public:
	e_Stream<e_BVHNodeData>* m_pNodes;
	e_Stream<float4x4>* m_pTransforms;
	e_Stream<float4x4>* m_pInvTransforms;
	int startNode;
	AABB m_sBox;
public:
	e_SceneBVH(unsigned int a_NodeCount);
	~e_SceneBVH();
	void Build(e_BufferReference<e_Node, e_Node>, e_BufferReference<e_Mesh, e_KernelMesh> a_Meshes);
	e_KernelSceneBVH getData(bool devicePointer = true);
	unsigned int getSizeInBytes();
	void setTransform(unsigned int nodeIdx, const float4x4& mat);
	void UpdateInvalidated();
private:

};
