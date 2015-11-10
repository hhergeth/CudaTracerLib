#pragma once

#include <MathTypes.h>
#include <Base/FixedSizeArray.h>

class e_Mesh;
struct e_KernelMaterial;
template<typename H, typename D> class e_BufferReference;

class e_Node
{
public:
	unsigned int m_uMeshIndex;
	unsigned int m_uMaterialOffset;
	unsigned int m_uInstanciatedMaterial;
	FixedSizeArray<unsigned int, MAX_AREALIGHT_NUM, true, 0xff> m_uLights;
public:
	e_Node() {}
	e_Node(unsigned int MeshIndex, e_Mesh* mesh, e_BufferReference<e_KernelMaterial, e_KernelMaterial> mat);
	AABB getWorldBox(e_Mesh* mesh, const float4x4& mat) const;
};