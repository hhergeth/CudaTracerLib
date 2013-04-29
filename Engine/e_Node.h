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
public:
	//host side data
	e_Mesh* m_pMesh;
	char m_cFile[256];
public:
	e_Node() {}
	e_Node(unsigned int MeshIndex, e_Mesh* mesh, const char* file)
	{
		m_pMesh = mesh;
		m_uMeshIndex = MeshIndex;
		m_sWorldMatrix = m_sInvWorldMatrix = float4x4::Identity();
		ZeroMemory(m_cFile, sizeof(m_cFile));
		strcpy(m_cFile, file);
		m_uMaterialOffset = mesh->m_sMatInfo.getIndex();
	}
	bool usesInstanciatedMaterials()
	{
		return m_uMaterialOffset != m_pMesh->m_sMatInfo.getIndex();
	}
	char* getFilePath()
	{
		return m_cFile;
	}
	void loadMesh(e_Mesh* m, unsigned int m2)
	{
		m_uMeshIndex = m2;
		m_pMesh = m;
	}
	AABB getWorldBox()
	{
		return m_pMesh->m_sLocalBox.Transform(m_sWorldMatrix);
	}
	CUDA_FUNC_IN float4x4 getWorldMatrix()
	{
		return m_sWorldMatrix;
	}
	CUDA_FUNC_IN float4x4 getInvWorldMatrix()
	{
		return m_sInvWorldMatrix;
	}
	void setTransform(float4x4& m)
	{
		m_sWorldMatrix = m;
		m_sInvWorldMatrix = m.Inverse();
	}
};