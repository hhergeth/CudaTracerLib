#pragma once

#include <MathTypes.h>
#include "e_TriangleData.h"
#include "e_Material.h"
#include "e_Buffer.h"
#include "..\Base\BVHBuilder.h"
#include "e_IntersectorData.h"

struct e_KernelMesh
{
	unsigned int m_uTriangleOffset;
	unsigned int m_uBVHNodeOffset;
	unsigned int m_uBVHTriangleOffset;
	unsigned int m_uBVHIndicesOffset;
	unsigned int m_uStdMaterialOffset;
};

#include "cuda_runtime.h"
#include "..\Base\FileStream.h"
#include "e_SceneInitData.h"

struct e_MeshPartLight
{
	FixedString<32> MatName;
	Spectrum L;
};

#define MESH_STATIC_TOKEN 1
#define MESH_ANIMAT_TOKEN 2

class e_Mesh
{
public:
	AABB m_sLocalBox;
	int m_uType;
public:
	e_StreamReference(e_TriangleData) m_sTriInfo;
	e_StreamReference(e_KernelMaterial) m_sMatInfo;
	e_StreamReference(e_BVHNodeData) m_sNodeInfo;
	e_StreamReference(e_TriIntersectorData) m_sIntInfo;
	e_StreamReference(e_TriIntersectorData2) m_sIndicesInfo;
	e_MeshPartLight m_sLights[MAX_AREALIGHT_NUM];
	unsigned int m_uUsedLights;
public:
	e_Mesh(IInStream& a_In, e_Stream<e_TriIntersectorData>* a_Stream0, e_Stream<e_TriangleData>* a_Stream1, e_Stream<e_BVHNodeData>* a_Stream2, e_Stream<e_TriIntersectorData2>* a_Stream3, e_Stream<e_KernelMaterial>* a_Stream4);
	void Free(e_Stream<e_TriIntersectorData>* a_Stream0, e_Stream<e_TriangleData>* a_Stream1, e_Stream<e_BVHNodeData>* a_Stream2, e_Stream<e_TriIntersectorData2>* a_Stream3, e_Stream<e_KernelMaterial>* a_Stream4);
	e_KernelMesh getKernelData()
	{
		e_KernelMesh m_sData;
		m_sData.m_uBVHIndicesOffset = m_sIndicesInfo.getIndex();
		m_sData.m_uBVHNodeOffset = m_sNodeInfo.getIndex() * sizeof(e_BVHNodeData) / sizeof(float4);
		m_sData.m_uBVHTriangleOffset = m_sIntInfo.getIndex() * 3;
		m_sData.m_uTriangleOffset = m_sTriInfo.getIndex();
		m_sData.m_uStdMaterialOffset = m_sMatInfo.getIndex();
		return m_sData;
	}
	unsigned int getTriangleCount()
	{
		return m_sTriInfo.getLength();
	}
	static void CompileMesh(const Vec3f* vertices, unsigned int nVertices, const Vec2f* uvs, const unsigned int* indices, unsigned int nIndices, const e_KernelMaterial& mat, const Spectrum& Le, OutputStream& out);
	static void CompileMesh(const Vec3f* vertices, unsigned int nVertices, const Vec2f** uvs, unsigned int nUV_Sets, const unsigned int* indices, unsigned int nIndices, const std::vector<e_KernelMaterial>& mats, const std::vector<Spectrum>& Le, const std::vector<unsigned int>& subMeshes, const unsigned char* extraData, OutputStream& out);
	static e_SceneInitData ParseBinary(const char* a_InputFile);
};
