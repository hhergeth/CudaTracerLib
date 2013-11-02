#pragma once

#include <MathTypes.h>
#include "e_Mesh.h"

class e_Node
{
public:
	unsigned int m_uMeshIndex;
	unsigned int m_uMaterialOffset;
	unsigned int m_uLightIndices[MAX_AREALIGHT_NUM];
#ifdef _DEBUG
public:
	char m_cFile[256];
#endif
public:
	e_Node() {}
	e_Node(unsigned int MeshIndex, e_Mesh* mesh, const char* file, e_StreamReference(e_KernelMaterial) mat)
	{
		m_uMeshIndex = MeshIndex;
#ifdef _DEBUG
		Platform::SetMemory(m_cFile, sizeof(m_cFile));
		strcpy(m_cFile, file);
#endif
		m_uMaterialOffset = mat.getIndex();
		for(unsigned int i = 0; i< mesh->m_sMatInfo.getLength(); i++)
			mat(i) = *mesh->m_sMatInfo(i);
	}
	void setLightData( unsigned int* li, unsigned int lic)
	{
		for(unsigned int i = 0; i < lic; i++)
			m_uLightIndices[i] = li[i];
		for(unsigned int i = lic; i < sizeof(m_uLightIndices) / sizeof(unsigned int); i++)
			m_uLightIndices[i] = 0xffffffff;
	}
	AABB getWorldBox(e_Mesh* mesh, const float4x4& mat) const
	{
		return mesh->m_sLocalBox.Transform(mat);
	}
	unsigned int getNextFreeLightIndex()
	{
		for(int i = 0; i < sizeof(m_uLightIndices) / sizeof(unsigned int); i++)
			if(m_uLightIndices[i] == 0xffffffff)
				return i;
		return 0xffffffff;
	}
};