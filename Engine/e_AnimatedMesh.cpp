#include "StdAfx.h"
#include "e_AnimatedMesh.h"
#include <MathTypes.h>
#include "../Base/Platform.h"

e_BufferReference<char, char> malloc_aligned(e_Stream<char>* stream, unsigned int a_Count, unsigned int a_Alignment)
{
	e_BufferReference<char, char> ref = stream->malloc(a_Count + a_Alignment * 2);
	uintptr_t ptr = (uintptr_t)ref.getDevice();
	unsigned int diff = ptr % a_Alignment, off = a_Alignment - diff;
	if (diff)
	{
		e_BufferReference<char, char> refFree = e_BufferReference<char, char>(stream, ref.getIndex(), off);
		//stream->dealloc(refFree);
		return e_BufferReference<char, char>(stream, ref.getIndex() + off, ref.getLength() - off);
	}
	else return ref;
}

e_AnimatedMesh::e_AnimatedMesh(const std::string& path, IInStream& a_In, e_Stream<e_TriIntersectorData>* a_Stream0, e_Stream<e_TriangleData>* a_Stream1, e_Stream<e_BVHNodeData>* a_Stream2, e_Stream<e_TriIntersectorData2>* a_Stream3, e_Stream<e_KernelMaterial>* a_Stream4, e_Stream<char>* a_Stream5)
	: e_Mesh(path, a_In, a_Stream0, a_Stream1, a_Stream2, a_Stream3, a_Stream4)
{
	m_uType = MESH_ANIMAT_TOKEN;
	a_In.Read(&k_Data, sizeof(k_Data));
	for(unsigned int i = 0; i < k_Data.m_uAnimCount; i++)
	{
		e_Animation A;
		A.deSerialize(a_In, a_Stream5);
		m_pAnimations.push_back(A);
	}
	m_sVertices = a_Stream5->malloc(sizeof(e_AnimatedVertex) * k_Data.m_uVertexCount);//malloc_aligned(a_Stream5, sizeof(e_AnimatedVertex) * k_Data.m_uVertexCount, 16);//
	m_sVertices.ReadFrom(a_In);
	m_sTriangles = a_Stream5->malloc(sizeof(uint3) * m_sTriInfo.getLength());//malloc_aligned(a_Stream5, sizeof(uint3) * m_sTriInfo.getLength(), 16);//
	m_sTriangles.ReadFrom(a_In);
	uint3* idx = ((uint3*)m_sTriangles().operator char *()) + (m_sTriInfo.getLength() - 2);
	a_Stream5->UpdateInvalidated();
	m_pBuilder = 0;
}

void e_AnimatedMesh::CreateNewMesh(e_AnimatedMesh* A, e_Stream<e_TriIntersectorData>* a_Stream0, e_Stream<e_TriangleData>* a_Stream1, e_Stream<e_BVHNodeData>* a_Stream2, e_Stream<e_TriIntersectorData2>* a_Stream3, e_Stream<e_KernelMaterial>* a_Stream4, e_Stream<char>* a_Stream5)
{
	A->m_uType = MESH_ANIMAT_TOKEN;
	A->m_uUsedLights = 0;
	A->m_sLocalBox = m_sLocalBox;
	A->m_sMatInfo = m_sMatInfo;
	A->m_sIndicesInfo = a_Stream3->malloc(m_sIndicesInfo, true);
	A->m_sTriInfo = a_Stream1->malloc(m_sTriInfo, true);
	A->m_sNodeInfo = a_Stream2->malloc(m_sNodeInfo, true);
	A->m_sIntInfo = a_Stream0->malloc(m_sIntInfo, true);
	A->m_pBuilder = 0;
	
	A->k_Data = k_Data;
	A->m_pAnimations = m_pAnimations;
	A->m_sVertices = m_sVertices;
	A->m_sTriangles = m_sTriangles;
}