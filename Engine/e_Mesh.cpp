#include "StdAfx.h"
#include "e_Mesh.h"
#define TS_DEC_FRAMEWORK
#include "..\Base\TangentSpace.h"
#undef TS_DEC_FRAMEWORK
#include "e_Volumes.h"
#include "e_Light.h"
#include "SceneBuilder\Importer.h"

e_Mesh::e_Mesh(IInStream& a_In, e_Stream<e_TriIntersectorData>* a_Stream0, e_Stream<e_TriangleData>* a_Stream1, e_Stream<e_BVHNodeData>* a_Stream2, e_Stream<e_TriIntersectorData2>* a_Stream3, e_Stream<e_KernelMaterial>* a_Stream4)
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
	//e_KernelMaterial*m = m_sMatInfo(4).operator e_KernelMaterial *();
//	std::cout << ((CudaVirtualAggregate<e_BaseType, e_BaseType>*)&m->bsdf)->getTypeToken() << "\n";
	m_sMatInfo.Invalidate();

	unsigned long long m_uNodeSize;
	a_In >> m_uNodeSize;
	m_sNodeInfo = a_Stream2->malloc(m_uNodeSize);
	a_In.Read(m_sNodeInfo(0), m_uNodeSize * sizeof(e_BVHNodeData));
	m_sNodeInfo.Invalidate();

	unsigned long long m_uIntSize;
	a_In >> m_uIntSize;
	m_sIntInfo = a_Stream0->malloc(m_uIntSize);
	a_In.Read(m_sIntInfo(0), m_uIntSize * sizeof(e_TriIntersectorData));
	m_sIntInfo.Invalidate();

	unsigned long long m_uIndicesSize;
	a_In >> m_uIndicesSize;
	m_sIndicesInfo = a_Stream3->malloc(m_uIndicesSize);
	a_In.Read(m_sIndicesInfo(0), m_uIndicesSize * sizeof(e_TriIntersectorData2));
	m_sIndicesInfo.Invalidate();
}

void e_Mesh::Free(e_Stream<e_TriIntersectorData>* a_Stream0, e_Stream<e_TriangleData>* a_Stream1, e_Stream<e_BVHNodeData>* a_Stream2, e_Stream<e_TriIntersectorData2>* a_Stream3, e_Stream<e_KernelMaterial>* a_Stream4)
{
	a_Stream0->dealloc(m_sIntInfo);
	a_Stream1->dealloc(m_sTriInfo);
	a_Stream2->dealloc(m_sNodeInfo);
	a_Stream3->dealloc(m_sIndicesInfo);
	a_Stream4->dealloc(m_sMatInfo);
}

e_SceneInitData e_Mesh::ParseBinary(const char* a_InputFile)
{
	InputStream a_In(a_InputFile);
	AABB m_sLocalBox;
	a_In >> m_sLocalBox;
	a_In.Move(sizeof(e_MeshPartLight) * MAX_AREALIGHT_NUM + 8);
#define PRINT(n, t) { a_In.Move(n * sizeof(t)); char msg[255]; msg[0] = 0; sprintf(msg, "Buf : %s, length : %d, size : %d[MB]\n", #t, (n), (n) * sizeof(t) / (1024 * 1024)); Platform::OutputDebug(msg);}
	unsigned int m_uTriangleCount;
	a_In >> m_uTriangleCount;
	PRINT(m_uTriangleCount, e_TriangleData)
	unsigned int m_uMaterialCount;
	a_In >> m_uMaterialCount;
	PRINT(m_uMaterialCount, e_KernelMaterial)
	unsigned long long m_uNodeSize;
	a_In >> m_uNodeSize;
	PRINT(m_uNodeSize, e_BVHNodeData)
	unsigned long long m_uIntSize;
	a_In >> m_uIntSize;
	PRINT(m_uIntSize, e_TriIntersectorData)
	unsigned long long m_uIndicesSize;
	a_In >> m_uIndicesSize;
	PRINT(m_uIndicesSize, e_TriIntersectorData2)
#undef PRINT
#undef PRINT2
	a_In.Close();
	char msg[2048];
	sprintf(msg, "return CreateForSpecificMesh(%d, %d, %d, %d, 255, a_Lights);\n", m_uTriangleCount, m_uIntSize, m_uNodeSize, m_uIndicesSize);
	Platform::OutputDebug(msg);
	return e_SceneInitData::CreateForSpecificMesh(m_uTriangleCount, m_uIntSize, m_uNodeSize, m_uIndicesSize, 255, 16, 16, 8);
}

void e_Mesh::CompileMesh(const Vec3f* vertices, unsigned int nVertices, const Vec2f* uvs, const unsigned int* indices, unsigned int nIndices, const e_KernelMaterial& mat, const Spectrum& Le, OutputStream& a_Out)
{
	e_MeshPartLight m_sLights[MAX_AREALIGHT_NUM];
	Platform::SetMemory(m_sLights, sizeof(m_sLights));
	unsigned int lightCount = 0;
	if(!Le.isZero())
	{
		m_sLights[0].L = Le;
		m_sLights[0].MatName = mat.Name;
		lightCount++;
	}
	Vec3f p[3];
	Vec3f n[3];
	Vec3f ta[3];
	Vec3f bi[3];
	Vec2f t[3];
	unsigned int numTriangles = indices ? nIndices / 3 : nVertices / 3;
	e_TriangleData* triData = new e_TriangleData[numTriangles];
	unsigned int triIndex = 0;
#ifdef EXT_TRI
	Vec3f* v_Normals = new Vec3f[nVertices], *v_Tangents = new Vec3f[nVertices], *v_BiTangents = new Vec3f[nVertices];
	Platform::SetMemory(v_Normals, sizeof(Vec3f) * nVertices);
	Platform::SetMemory(v_Tangents, sizeof(Vec3f) * nVertices);
	Platform::SetMemory(v_BiTangents, sizeof(Vec3f) * nVertices);
	ComputeTangentSpace(vertices, uvs, indices, nVertices, numTriangles, v_Normals, v_Tangents, v_BiTangents);
#endif
	AABB box = AABB::Identity();
	for(size_t ti = 0; ti < numTriangles; ti++)
	{
		for(size_t j = 0; j < 3; j++)
		{
			size_t l = indices ? indices[ti * 3 + j] : ti * 3 + j;
			p[j] = vertices[l];
			box.Enlarge(p[j]);
#ifdef EXT_TRI
			if(uvs)
				t[j] = uvs[l];
			ta[j] = normalize(v_Tangents[l]);
			bi[j] = normalize(v_BiTangents[l]);
			n[j] = normalize(v_Normals[l]);
#endif
		}
		triData[triIndex++] = e_TriangleData(p, 0, t, n, ta, bi);
	}
	a_Out << box;
	a_Out.Write(m_sLights, sizeof(m_sLights));
	a_Out << lightCount;
	a_Out << numTriangles;
	a_Out.Write(triData, sizeof(e_TriangleData) * numTriangles);
	a_Out << (unsigned int)1;
	a_Out.Write(&mat, sizeof(e_KernelMaterial));
	ConstructBVH(vertices, indices, nVertices, numTriangles * 3, a_Out);
#ifdef EXT_TRI
	delete [] v_Normals;
	delete [] v_Tangents;
	delete [] v_BiTangents;
#endif
	delete [] triData;
}

void e_Mesh::CompileMesh(const Vec3f* vertices, unsigned int nVertices, const Vec2f** uvs, unsigned int nUV_Sets, const unsigned int* indices, unsigned int nIndices, const std::vector<e_KernelMaterial>& mats, const std::vector<Spectrum>& Les, const std::vector<unsigned int>& subMeshes, const unsigned char* extraData, OutputStream& a_Out)
{
	e_MeshPartLight m_sLights[MAX_AREALIGHT_NUM];
	Platform::SetMemory(m_sLights, sizeof(m_sLights));
	unsigned int lightCount = 0;
	if(!Les[0].isZero())
	{
		m_sLights[lightCount].L = Les[0];
		m_sLights[0].MatName = mats[0].Name;
		lightCount++;
	}
	Vec3f p[3];
	Vec3f n[3];
	Vec3f ta[3];
	Vec3f bi[3];
	Vec2f t[3];
	unsigned int numTriangles = indices ? nIndices / 3 : nVertices / 3;
	e_TriangleData* triData = new e_TriangleData[numTriangles];
	unsigned int triIndex = 0;
#ifdef EXT_TRI
	Vec3f* v_Normals = new Vec3f[nVertices], *v_Tangents = new Vec3f[nVertices], *v_BiTangents = new Vec3f[nVertices];
	Platform::SetMemory(v_Normals, sizeof(Vec3f) * nVertices);
	Platform::SetMemory(v_Tangents, sizeof(Vec3f) * nVertices);
	Platform::SetMemory(v_BiTangents, sizeof(Vec3f) * nVertices);
	//compute the frame for the first set and hope the rest is aligned
	ComputeTangentSpace(vertices, uvs[0], indices, nVertices, numTriangles, v_Normals, v_Tangents, v_BiTangents);
#endif
	AABB box = AABB::Identity();
	unsigned int si = 0, pc = 0;
	for(size_t ti = 0; ti < numTriangles; ti++)
	{
		for(size_t j = 0; j < 3; j++)
		{
			size_t l = indices ? indices[ti * 3 + j] : ti * 3 + j;
			p[j] = vertices[l];
			box.Enlarge(p[j]);
#ifdef EXT_TRI
			ta[j] = normalize(v_Tangents[l]);
			bi[j] = normalize(v_BiTangents[l]);
			n[j] = normalize(v_Normals[l]);
#endif
		}
		e_TriangleData tri(p, (unsigned char)si, t, n, ta, bi);
		for(unsigned int uvIdx = 0; uvIdx < nUV_Sets; uvIdx++)
		{
			for(int j = 0; j < 3; j++)
			{
				size_t l = indices ? indices[ti * 3 + j] : ti * 3 + j;
				t[j] = uvs[uvIdx][l];
			}
			tri.setUvSetData(uvIdx, t[0], t[1], t[2]);
		}
		if(extraData)
			for(int j = 0; j < 3; j++)
				tri.m_sHostData.ExtraData = extraData[indices ? indices[ti * 3 + j] : ti * 3 + j];
		triData[triIndex++] = tri;
		if(subMeshes[si] + pc <= ti)
		{
			pc += subMeshes[si];
			si++;
			if(!Les[si].isZero())
			{
				m_sLights[lightCount].L = Les[si];
				m_sLights[lightCount].MatName = mats[si].Name;
				lightCount++;
			}
		}
	}
	a_Out << box;
	a_Out.Write(m_sLights, sizeof(m_sLights));
	a_Out << lightCount;
	a_Out << numTriangles;
	a_Out.Write(triData, sizeof(e_TriangleData) * numTriangles);
	a_Out << (unsigned int)mats.size();
	a_Out.Write(&mats[0], sizeof(e_KernelMaterial) * (unsigned int)mats.size());
	ConstructBVH(vertices, indices, nVertices, numTriangles * 3, a_Out);
#ifdef EXT_TRI
	delete [] v_Normals;
	delete [] v_Tangents;
	delete [] v_BiTangents;
#endif
	delete [] triData;
}