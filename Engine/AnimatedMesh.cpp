#include "StdAfx.h"
#include "AnimatedMesh.h"
#include "Buffer.h"
#include <Math/Vector.h>
#include <Base/FileStream.h>
#include "TriangleData.h"
#include "Material.h"
#include "TriIntersectorData.h"

namespace CudaTracerLib {

void AnimationFrame::serialize(FileOutputStream& a_Out)
{
	a_Out << (size_t)m_sHostConstructionData.size();
	a_Out.Write(&m_sHostConstructionData[0], sizeof(float4x4) * (unsigned int)m_sHostConstructionData.size());
}

void AnimationFrame::deSerialize(IInStream& a_In, Stream<char>* Buf)
{
	size_t A;
	a_In >> A;
	m_sMatrices = Buf->malloc_aligned(sizeof(float4x4) * (unsigned int)A, 16);
	a_In >> m_sMatrices;
}

void Animation::serialize(FileOutputStream& a_Out)
{
	a_Out << (size_t)m_pFrames.size();
	a_Out << m_uFrameRate;
	a_Out.Write(m_sName);
	for (unsigned int i = 0; i < m_pFrames.size(); i++)
		m_pFrames[i].serialize(a_Out);
}

void Animation::deSerialize(IInStream& a_In, Stream<char>* Buf)
{
	size_t m_uNumFrames;
	a_In >> m_uNumFrames;
	a_In >> m_uFrameRate;
	a_In.Read(m_sName);
	for (unsigned int i = 0; i < m_uNumFrames; i++)
	{
		m_pFrames.push_back(AnimationFrame());
		m_pFrames[i].deSerialize(a_In, Buf);
	}
}

AnimatedMesh::AnimatedMesh(const std::string& path, IInStream& a_In, Stream<TriIntersectorData>* a_Stream0, Stream<TriangleData>* a_Stream1, Stream<BVHNodeData>* a_Stream2, Stream<TriIntersectorData2>* a_Stream3, Stream<Material>* a_Stream4, Stream<char>* a_Stream5)
	: Mesh(path, a_In, a_Stream0, a_Stream1, a_Stream2, a_Stream3, a_Stream4, a_Stream5)
{
	m_uType = MESH_ANIMAT_TOKEN;
	a_In.Read(&k_Data, sizeof(k_Data));
	for (unsigned int i = 0; i < k_Data.m_uAnimCount; i++)
	{
		Animation A;
		A.deSerialize(a_In, a_Stream5);
		m_pAnimations.push_back(A);
	}
	m_sVertices = a_Stream5->malloc_aligned<AnimatedVertex>(sizeof(AnimatedVertex) * k_Data.m_uVertexCount);
	a_In >> m_sVertices;
	m_sTriangles = a_Stream5->malloc_aligned<uint3>(sizeof(uint3) * m_sTriInfo.getLength());
	a_In >> m_sTriangles;
	a_Stream5->UpdateInvalidated();
	m_pBuilder = 0;
}

void AnimatedMesh::CreateNewMesh(AnimatedMesh* A, Stream<TriIntersectorData>* a_Stream0, Stream<TriangleData>* a_Stream1, Stream<BVHNodeData>* a_Stream2, Stream<TriIntersectorData2>* a_Stream3, Stream<Material>* a_Stream4, Stream<char>* a_Stream5) const
{
	A->m_uType = MESH_ANIMAT_TOKEN;
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

}