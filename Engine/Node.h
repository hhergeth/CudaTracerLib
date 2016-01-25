#pragma once

#include <Base/FixedSizeArray.h>

namespace CudaTracerLib {

struct AABB;
struct float4x4;
class Mesh;
struct Material;
template<typename H, typename D> class BufferReference;

class Node
{
public:
	unsigned int m_uMeshIndex;
	unsigned int m_uMaterialOffset;
	unsigned int m_uInstanciatedMaterial;
	FixedSizeArray<unsigned int, MAX_AREALIGHT_NUM, true, 0xff> m_uLights;
public:
	Node() {}
	Node(unsigned int MeshIndex, Mesh* mesh, BufferReference<Material, Material> mat);
	AABB getWorldBox(Mesh* mesh, const float4x4& mat) const;
};

}