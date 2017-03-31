#pragma once

#include <Math/float4x4.h>
#include <Math/AABB.h>
#include "SceneBVH_device.h"
#include <Base/Buffer_device.h>

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
	CTL_EXPORT SceneBVH(size_t a_NodeCount);
	CTL_EXPORT ~SceneBVH();
	CTL_EXPORT bool Build(Stream<Node>* nodStream, Buffer<Mesh, KernelMesh>* mesh_buf);
	CTL_EXPORT KernelSceneBVH getData(bool devicePointer = true);
	CTL_EXPORT size_t getDeviceSizeInBytes();
	CTL_EXPORT void setTransform(BufferReference<Node, Node> n, const float4x4& mat);
	CTL_EXPORT void invalidateNode(BufferReference<Node, Node> n);
	CTL_EXPORT void addNode(BufferReference<Node, Node> n);
	CTL_EXPORT void removeNode(BufferReference<Node, Node> n);
	CTL_EXPORT const float4x4& getNodeTransform(BufferReference<Node, Node> n);
	CTL_EXPORT BVHNodeData* getBVHNode(unsigned int i);
	CTL_EXPORT bool needsBuild();
	CTL_EXPORT AABB getSceneBox();
	CTL_EXPORT void printGraph(const std::string& path);
};

}