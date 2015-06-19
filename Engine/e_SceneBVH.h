#pragma once

#include <MathTypes.h>
#include "e_SceneBVH_device.h"
#include <set>
#include "e_Buffer_device.h"

class e_Node;
class e_Mesh;
struct e_KernelMesh;
template<typename T> class e_Stream;
template<typename H, typename D> class e_Buffer;
template<typename H, typename D> class e_BufferRange;
class e_BVHRebuilder;

class e_SceneBVH
{
private:
	e_Stream<e_BVHNodeData>* m_pNodes;
	e_Stream<float4x4>* m_pTransforms;
	e_Stream<float4x4>* m_pInvTransforms;
	e_BVHRebuilder* m_pBuilder;
	e_BufferReference<e_BVHNodeData, e_BVHNodeData> node_ref;
	e_BufferReference<float4x4, float4x4> tr_ref, iv_tr_ref;
public:
	e_SceneBVH(size_t a_NodeCount);
	~e_SceneBVH();
	bool Build(e_Stream<e_Node>* node_stream, e_Buffer<e_Mesh, e_KernelMesh>* mesh_buf);
	e_KernelSceneBVH getData(bool devicePointer = true);
	size_t getDeviceSizeInBytes();
	void setTransform(e_BufferReference<e_Node, e_Node> n, const float4x4& mat);
	void invalidateNode(e_BufferReference<e_Node, e_Node> n);
	void addNode(e_BufferReference<e_Node, e_Node> n);
	void removeNode(e_BufferReference<e_Node, e_Node> n);
	const float4x4& getNodeTransform(e_BufferReference<e_Node, e_Node> n);
	e_BVHNodeData* getBVHNode(unsigned int i);
	bool needsBuild();
	AABB getSceneBox();
	void printGraph(const std::string& path);
};
