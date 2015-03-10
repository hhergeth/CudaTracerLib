#include <StdAfx.h>
#include "e_Node.h"
#include "e_Mesh.h"

e_Node::e_Node(unsigned int MeshIndex, e_Mesh* mesh, const char* file, e_StreamReference(e_KernelMaterial) mat)
{
	m_uMeshIndex = MeshIndex;
#ifdef _DEBUG
	Platform::SetMemory(m_cFile, sizeof(m_cFile));
	strcpy(m_cFile, file);
#endif
	m_uMaterialOffset = mat.getIndex();
	for (unsigned int i = 0; i< mesh->m_sMatInfo.getLength(); i++)
		mat(i) = *mesh->m_sMatInfo(i);
	for (int i = 0; i < MAX_AREALIGHT_NUM; i++)
		m_uLightIndices[i] = 0xffffffff;
}

AABB e_Node::getWorldBox(e_Mesh* mesh, const float4x4& mat) const
{
	return mesh->m_sLocalBox.Transform(mat);
}