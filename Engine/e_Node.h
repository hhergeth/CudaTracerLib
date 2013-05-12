#pragma once

#include "..\Math\vector.h"
#include "e_Mesh.h"

class e_Node
{
private:
	//kernel side data
	float4x4 m_sWorldMatrix;
	float4x4 m_sInvWorldMatrix;
public:
	unsigned int m_uMeshIndex;
	unsigned int m_uMaterialOffset;
	unsigned int m_uLightIndices[MAX_AREALIGHT_NUM];
public:
	char m_cFile[256];
public:
	e_Node() {}
	e_Node(unsigned int MeshIndex, e_Mesh* mesh, const char* file)
	{
		m_uMeshIndex = MeshIndex;
		m_sWorldMatrix = m_sInvWorldMatrix = float4x4::Identity();
		ZeroMemory(m_cFile, sizeof(m_cFile));
		strcpy(m_cFile, file);
		m_uMaterialOffset = mesh->m_sMatInfo.getIndex();
	}
	void setLightData( unsigned int* li, unsigned int lic)
	{
		for(int i = 0; i < lic; i++)
			m_uLightIndices[i] = li[i];
		for(int i = lic; i < sizeof(m_uLightIndices) / sizeof(unsigned int); i++)
			m_uLightIndices[i] = -1;
	}
	bool usesInstanciatedMaterials(e_Mesh* mesh) const
	{
		return m_uMaterialOffset != mesh->m_sMatInfo.getIndex();
	}
	const char* getFilePath() const
	{
		return m_cFile;
	}/*
	void loadMesh(e_Mesh* m, unsigned int m2)
	{
		m_uMeshIndex = m2;
		m_pMesh = m;
	}*/
	AABB getWorldBox(e_Mesh* mesh) const
	{
		return mesh->m_sLocalBox.Transform(m_sWorldMatrix);
	}
	CUDA_FUNC_IN float4x4 getWorldMatrix() const
	{
		return m_sWorldMatrix;
	}
	CUDA_FUNC_IN float4x4 getInvWorldMatrix() const
	{
		return m_sInvWorldMatrix;
	}
	void setTransform(const float4x4& m)
	{
		m_sWorldMatrix = m;
		m_sInvWorldMatrix = m.Inverse();
	}
	unsigned int getNextFreeLightIndex()
	{
		for(int i = 0; i < sizeof(m_uLightIndices) / sizeof(unsigned int); i++)
			if(m_uLightIndices[i] != -1)
				return i;
		return -1;
	}
};