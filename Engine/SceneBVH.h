#pragma once

#include <MathTypes.h>
#include "SceneBVH_device.h"
#include <set>
#include "Buffer_device.h"

namespace CudaTracerLib {

class Node;
class Mesh;
struct KernelMesh;
template<typename T> class Stream;
template<typename H, typename D> class Buffer;
template<typename H, typename D> class BufferRange;
class BVHRebuilder;

class SceneBVH
{
private:
	Stream<BVHNodeData>* m_pNodes;
	Stream<float4x4>* m_pTransforms;
	Stream<float4x4>* m_pInvTransforms;
	BVHRebuilder* m_pBuilder;
	BufferReference<BVHNodeData, BVHNodeData> node_ref;
	BufferReference<float4x4, float4x4> tr_ref, iv_tr_ref;
public:
	SceneBVH(size_t a_NodeCount);
	~SceneBVH();
	bool Build(Stream<Node>* nodStream, Buffer<Mesh, KernelMesh>* mesh_buf);
	KernelSceneBVH getData(bool devicePointer = true);
	size_t getDeviceSizeInBytes();
	void setTransform(BufferReference<Node, Node> n, const float4x4& mat);
	void invalidateNode(BufferReference<Node, Node> n);
	void addNode(BufferReference<Node, Node> n);
	void removeNode(BufferReference<Node, Node> n);
	const float4x4& getNodeTransform(BufferReference<Node, Node> n);
	BVHNodeData* getBVHNode(unsigned int i);
	bool needsBuild();
	AABB getSceneBox();
	void printGraph(const std::string& path);
};

}