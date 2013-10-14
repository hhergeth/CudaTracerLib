#pragma once

#include <MathTypes.h>
#include "e_Mesh.h"

class CUDA_ALIGN(16) e_Node
{
private:
	//kernel side data
	float4x4 m_sWorldMatrix;
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
		m_sWorldMatrix = float4x4::Identity();
#ifdef _DEBUG
		ZeroMemory(m_cFile, sizeof(m_cFile));
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
	AABB getWorldBox(e_Mesh* mesh) const
	{
		return mesh->m_sLocalBox.Transform(m_sWorldMatrix);
	}
	CUDA_FUNC_IN float4x4 getWorldMatrix() const
	{
		return m_sWorldMatrix;
	}
	void setTransform(const float4x4& m)
	{
		m_sWorldMatrix = m;
	}
	unsigned int getNextFreeLightIndex()
	{
		for(int i = 0; i < sizeof(m_uLightIndices) / sizeof(unsigned int); i++)
			if(m_uLightIndices[i] == 0xffffffff)
				return i;
		return 0xffffffff;
	}
};