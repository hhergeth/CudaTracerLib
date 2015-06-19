#pragma once

#include <MathTypes.h>

class e_Mesh;
struct e_KernelMaterial;
template<typename H, typename D> class e_BufferReference;

class e_Node
{
public:
	unsigned int m_uMeshIndex;
	unsigned int m_uMaterialOffset;
	unsigned int m_uInstanciatedMaterial;
	unsigned int m_uLightIndices[MAX_AREALIGHT_NUM];
public:
	e_Node() {}
	e_Node(unsigned int MeshIndex, e_Mesh* mesh, e_BufferReference<e_KernelMaterial, e_KernelMaterial> mat);
	void setLightData( unsigned int* li, unsigned int lic)
	{
		for(unsigned int i = 0; i < lic; i++)
			m_uLightIndices[i] = li[i];
		for(unsigned int i = lic; i < sizeof(m_uLightIndices) / sizeof(unsigned int); i++)
			m_uLightIndices[i] = 0xffffffff;
	}
	AABB getWorldBox(e_Mesh* mesh, const float4x4& mat) const;
	unsigned int getNextFreeLightIndex()
	{
		for(int i = 0; i < sizeof(m_uLightIndices) / sizeof(unsigned int); i++)
			if(m_uLightIndices[i] == 0xffffffff)
				return i;
		return 0xffffffff;
	}
};