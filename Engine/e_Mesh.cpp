#include "StdAfx.h"
#include "e_Mesh.h"
#define TS_DEC_FRAMEWORK
#include "..\Base\TangentSpace.h"
#undef TS_DEC_FRAMEWORK
#include "e_Volumes.h"
#include "e_Light.h"
#include "SceneBuilder\Importer.h"

e_Mesh::e_Mesh(InputStream& a_In, e_Stream<e_TriIntersectorData>* a_Stream0, e_Stream<e_TriangleData>* a_Stream1, e_Stream<e_BVHNodeData>* a_Stream2, e_Stream<int>* a_Stream3, e_Stream<e_KernelMaterial>* a_Stream4)
{
	m_uType = MESH_STATIC_TOKEN;
	int abc = sizeof(e_TriangleData), abc2 = sizeof(m_sLights);

	a_In >> m_sLocalBox;
	a_In.Read(m_sLights, sizeof(m_sLights));
	a_In >> m_uUsedLights;

	unsigned int m_uTriangleCount;
	a_In >> m_uTriangleCount;
	m_sTriInfo = a_Stream1->malloc(m_uTriangleCount);
	a_In.Read(m_sTriInfo(0), m_sTriInfo.getSizeInBytes());
	m_sTriInfo.Invalidate();

	unsigned int m_uMaterialCount;
	a_In >> m_uMaterialCount;
	m_sMatInfo = a_Stream4->malloc(m_uMaterialCount);
	a_In.Read(m_sMatInfo(0), m_sMatInfo.getSizeInBytes());
	m_sMatInfo.Invalidate();

	a_In >> m_uMaterialCount;
	unsigned long long m_uNodeSize;
	a_In >> m_uNodeSize;
	m_sNodeInfo = a_Stream2->malloc(m_uNodeSize / 64);
	e_BVHNodeData* n2 = m_sNodeInfo(0);
	a_In.Read(n2, m_uNodeSize);
	m_sNodeInfo.Invalidate();

	unsigned long long m_uIntSize;
	a_In >> m_uIntSize;
	float C = ceil((float)m_uIntSize / 48.0f);
	m_sIntInfo = a_Stream0->malloc((int)C);
	a_In.Read(m_sIntInfo(0), m_uIntSize);
	m_sIntInfo.Invalidate();

	unsigned long long m_uIndicesSize;
	a_In >> m_uIndicesSize;
	m_sIndicesInfo = a_Stream3->malloc(m_uIndicesSize / 4);
	a_In.Read(m_sIndicesInfo(0), m_sIndicesInfo.getSizeInBytes());
	m_sIndicesInfo.Invalidate();
}

e_SceneInitData e_Mesh::ParseBinary(const char* a_InputFile)
{
	InputStream a_In(a_InputFile);
	AABB m_sLocalBox;
	a_In >> m_sLocalBox;
	a_In.Move(sizeof(e_MeshPartLight) * MAX_AREALIGHT_NUM + 8);
#define PRINT(n, t) { a_In.Move(n * sizeof(t)); char msg[255]; msg[0] = 0; sprintf(msg, "Buf : %s, length : %d, size : %d[MB]\n", #t, (n), (n) * sizeof(t) / (1024 * 1024)); Platform::OutputDebug(msg);}
#define PRINT2(n, t) { a_In.Move(n); char msg[255]; msg[0] = 0; sprintf(msg, "Buf : %s, length : %d, size : %d[MB]\n", #t, (n) / sizeof(t), (n) / (1024 * 1024)); Platform::OutputDebug(msg);}
	unsigned int m_uTriangleCount;
	a_In >> m_uTriangleCount;
	PRINT(m_uTriangleCount, e_TriangleData)
	unsigned int m_uMaterialCount;
	a_In >> m_uMaterialCount;
	PRINT(m_uMaterialCount, e_KernelMaterial)
	a_In >> m_uMaterialCount;
	unsigned long long m_uNodeSize;
	a_In >> m_uNodeSize;
	PRINT(m_uNodeSize / 64, e_BVHNodeData)
	unsigned long long m_uIntSize;
	a_In >> m_uIntSize;
	PRINT2(m_uIntSize, e_TriIntersectorData)
	unsigned long long m_uIndicesSize;
	a_In >> m_uIndicesSize;
	PRINT(m_uIndicesSize / 4, int)
#undef PRINT
#undef PRINT2
	a_In.Close();
	char msg[2048];
	sprintf(msg, "return CreateForSpecificMesh(%d, %d, %d, %d, 255, a_Lights);\n", m_uTriangleCount, (int)ceilf((float)m_uIntSize / 48.0f), m_uNodeSize / 64, m_uIndicesSize / 4);
	Platform::OutputDebug(msg);
	return e_SceneInitData::CreateForSpecificMesh(m_uTriangleCount, (int)ceilf((float)m_uIntSize / 48.0f), m_uNodeSize / 64, m_uIndicesSize / 4, 255, 16, 16, 8);
}

void e_Mesh::Free(e_Stream<e_TriIntersectorData>& a_Stream0, e_Stream<e_TriangleData>& a_Stream1, e_Stream<e_BVHNodeData>& a_Stream2, e_Stream<int>& a_Stream3, e_Stream<e_KernelMaterial>& a_Stream4)
{
	a_Stream0.dealloc(m_sIntInfo);
	a_Stream1.dealloc(m_sTriInfo);
	a_Stream2.dealloc(m_sNodeInfo);
	a_Stream3.dealloc(m_sIndicesInfo);
	a_Stream4.dealloc(m_sMatInfo);
}