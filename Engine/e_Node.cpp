#include <StdAfx.h>
#include "e_Node.h"
#include "e_Mesh.h"
#include "e_Buffer.h"
#include "e_Material.h"
#include "e_TriIntersectorData.h"

namespace CudaTracerLib {

e_Node::e_Node(unsigned int MeshIndex, e_Mesh* mesh, e_StreamReference<e_KernelMaterial> mat)
	: m_uInstanciatedMaterial(false)
{
	m_uMeshIndex = MeshIndex;
	m_uMaterialOffset = mat.getIndex();
	for (unsigned int i = 0; i < mesh->m_sMatInfo.getLength(); i++)
		mat(i) = *mesh->m_sMatInfo(i);
}

AABB e_Node::getWorldBox(e_Mesh* mesh, const float4x4& mat) const
{
	return mesh->m_sLocalBox.Transform(mat);
}

}