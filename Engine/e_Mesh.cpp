#include "StdAfx.h"
#include "e_Mesh.h"
#include "SceneBuilder/TangentSpaceHelper.h"
#include "e_Volumes.h"
#include "e_Light.h"
#include "SceneBuilder/Importer.h"
#include "e_Buffer.h"
#include <Base/FileStream.h>
#include "e_SceneInitData.h"
#include "e_TriangleData.h"
#include "e_Material.h"
#include "e_TriIntersectorData.h"

namespace CudaTracerLib {

#define NO_NODE 0x76543210
void write(int idx, const e_BVHNodeData* nodes, int parent, std::ofstream& f, int& leafC)
{
	if (parent != -1)
		f << parent << " -> " << idx << ";\n";
	Vec2i c = nodes[idx].getChildren();
	int p = *(int*)&nodes[idx].d.z;
	if (p != -1 && p != parent * 4)
		throw std::runtime_error("");
	if (p != -1)
		f << idx << " -> " << p / 4 << ";\n";
	if (c.x > 0 && c.x != NO_NODE)
		write(c.x / 4, nodes, idx, f, leafC);
	else if (c.x < 0)
		f << idx << " -> " << c.x << "[style=dotted];\n";
	if (c.y > 0 && c.y != NO_NODE)
		write(c.y / 4, nodes, idx, f, leafC);
	else if (c.y < 0)
		f << idx << " -> " << c.y << "[style=dotted];\n";
}
void printBVHData(const e_BVHNodeData* bvh, const std::string& path)
{
	std::ofstream file;
	file.open(path);
	file << "digraph SceneBVH {\nnode [fontname=\"Arial\"];\n";
	int leafC = 0;
	write(0, bvh, -1, file, leafC);
	file << "}";
	file.close();
}

e_Mesh::e_Mesh(const std::string& path, IInStream& a_In, e_Stream<e_TriIntersectorData>* a_Stream0, e_Stream<e_TriangleData>* a_Stream1, e_Stream<e_BVHNodeData>* a_Stream2, e_Stream<e_TriIntersectorData2>* a_Stream3, e_Stream<e_KernelMaterial>* a_Stream4, e_Stream<char>* a_Stream5)
	: m_uPath(path)
{
	m_uType = MESH_STATIC_TOKEN;

	a_In >> m_sLocalBox;
	unsigned int numLights;
	a_In >> numLights;
	m_sAreaLights.resize(numLights);
	if (numLights)
		a_In.Read(&m_sAreaLights[0], numLights * sizeof(e_MeshPartLight));

	unsigned int m_uTriangleCount;
	a_In >> m_uTriangleCount;
	m_sTriInfo = a_Stream1->malloc(m_uTriangleCount);
	a_In >> m_sTriInfo;
	m_sTriInfo.Invalidate();

	unsigned int m_uMaterialCount;
	a_In >> m_uMaterialCount;
	m_sMatInfo = a_Stream4->malloc(m_uMaterialCount);
	a_In >> m_sMatInfo;
	for (unsigned int i = 0; i < m_uMaterialCount; i++)
		m_sMatInfo(i)->bsdf.SetVtable();
	m_sMatInfo.Invalidate();

	unsigned long long m_uNodeSize;
	a_In >> m_uNodeSize;
	m_sNodeInfo = a_Stream2->malloc(m_uNodeSize);
	a_In >> m_sNodeInfo;
	m_sNodeInfo.Invalidate();

	unsigned long long m_uIntSize;
	a_In >> m_uIntSize;
	m_sIntInfo = a_Stream0->malloc(m_uIntSize);
	a_In >> m_sIntInfo;
	m_sIntInfo.Invalidate();

	unsigned long long m_uIndicesSize;
	a_In >> m_uIndicesSize;
	m_sIndicesInfo = a_Stream3->malloc(m_uIndicesSize);
	a_In >> m_sIndicesInfo;
	m_sIndicesInfo.Invalidate();

	//printBVHData(m_sNodeInfo(0), "mesh.txt");
}

e_KernelMesh e_Mesh::getKernelData()
{
	e_KernelMesh m_sData;
	m_sData.m_uBVHIndicesOffset = m_sIndicesInfo.getIndex();
	m_sData.m_uBVHNodeOffset = m_sNodeInfo.getIndex() * sizeof(e_BVHNodeData) / sizeof(float4);
	m_sData.m_uBVHTriangleOffset = m_sIntInfo.getIndex() * 3;
	m_sData.m_uTriangleOffset = m_sTriInfo.getIndex();
	m_sData.m_uStdMaterialOffset = m_sMatInfo.getIndex();
	return m_sData;
}

void e_Mesh::Free(e_Stream<e_TriIntersectorData>* a_Stream0, e_Stream<e_TriangleData>* a_Stream1, e_Stream<e_BVHNodeData>* a_Stream2, e_Stream<e_TriIntersectorData2>* a_Stream3, e_Stream<e_KernelMaterial>* a_Stream4)
{
	a_Stream0->dealloc(m_sIntInfo);
	a_Stream1->dealloc(m_sTriInfo);
	a_Stream2->dealloc(m_sNodeInfo);
	a_Stream3->dealloc(m_sIndicesInfo);
	a_Stream4->dealloc(m_sMatInfo);
}

e_SceneInitData e_Mesh::ParseBinary(const std::string& a_InputFile)
{
	FileInputStream a_In(a_InputFile);
	AABB m_sLocalBox;
	a_In >> m_sLocalBox;
	unsigned int numLights;
	a_In >> numLights;
	a_In.Move(sizeof(e_MeshPartLight) * numLights);
#define PRINT(n, t) { a_In.Move(n * sizeof(t)); Platform::OutputDebug(format("Buf : %s, length : %llu, size : %llu[MB]\n", #t, size_t(n), size_t((n) * sizeof(t) / (1024 * 1024))));}
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
	Platform::OutputDebug(format("return CreateForSpecificMesh(%d, %d, %d, %d, 255, a_Lights);\n", m_uTriangleCount, m_uIntSize, m_uNodeSize, m_uIndicesSize));
	return e_SceneInitData::CreateForSpecificMesh(m_uTriangleCount, m_uIntSize, m_uNodeSize, m_uIndicesSize, 255, 16, 16, 8);
}

void e_Mesh::CompileMesh(const Vec3f* vertices, unsigned int nVertices, const Vec2f* uvs, const unsigned int* indices, unsigned int nIndices, const e_KernelMaterial& mat, const Spectrum& Le, FileOutputStream& a_Out)
{
	std::vector<e_MeshPartLight> lights;
	if (!Le.isZero())
		lights.push_back(e_MeshPartLight(mat.Name, Le));
	Vec3f p[3];
	Vec3f n[3];
	Vec3f ta[3];
	Vec3f bi[3];
	Vec2f t[3];
	t[0] = t[1] = t[2] = Vec2f(0.0f);
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
	for (size_t ti = 0; ti < numTriangles; ti++)
	{
		for (size_t j = 0; j < 3; j++)
		{
			size_t l = indices ? indices[ti * 3 + j] : ti * 3 + j;
			p[j] = vertices[l];
			box = box.Extend(p[j]);
#ifdef EXT_TRI
			if (uvs)
				t[j] = uvs[l];
			ta[j] = normalize(v_Tangents[l]);
			bi[j] = normalize(v_BiTangents[l]);
			n[j] = normalize(v_Normals[l]);
#endif
		}
		triData[triIndex++] = e_TriangleData(p, 0, t, n, ta, bi);
	}
	a_Out << box;
	a_Out << (unsigned int)lights.size();
	if (lights.size())
		a_Out.Write(&lights[0], lights.size() * sizeof(e_MeshPartLight));
	a_Out << numTriangles;
	a_Out.Write(triData, sizeof(e_TriangleData) * numTriangles);
	a_Out << (unsigned int)1;
	a_Out.Write(&mat, sizeof(e_KernelMaterial));
	ConstructBVH(vertices, indices, nVertices, numTriangles * 3, a_Out);
#ifdef EXT_TRI
	delete[] v_Normals;
	delete[] v_Tangents;
	delete[] v_BiTangents;
#endif
	delete[] triData;
}

void e_Mesh::CompileMesh(const Vec3f* vertices, unsigned int nVertices, const Vec2f** uvs, unsigned int nUV_Sets, const unsigned int* indices, unsigned int nIndices, const std::vector<e_KernelMaterial>& mats, const std::vector<Spectrum>& Les, const std::vector<unsigned int>& subMeshes, const unsigned char* extraData, FileOutputStream& a_Out)
{
	std::vector<e_MeshPartLight> lights;
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
	for (size_t ti = 0; ti < numTriangles; ti++)
	{
		e_TriangleData tri;
		for (unsigned int uvIdx = 0; uvIdx < Dmin2(nUV_Sets, NUM_UV_SETS); uvIdx++)
		{
			for (int j = 0; j < 3; j++)
			{
				size_t l = indices ? indices[ti * 3 + j] : ti * 3 + j;
				t[j] = uvs[uvIdx][l];
			}
			tri.setUvSetData(uvIdx, t[0], t[1], t[2]);
		}
		for (size_t j = 0; j < 3; j++)
		{
			size_t l = indices ? indices[ti * 3 + j] : ti * 3 + j;
			p[j] = vertices[l];
			box = box.Extend(p[j]);
#ifdef EXT_TRI
			ta[j] = normalize(v_Tangents[l]);
			bi[j] = normalize(v_BiTangents[l]);
			n[j] = normalize(v_Normals[l]);
#endif
		}
		tri.setData(p[0], p[1], p[2], n[0], n[1], n[2]);

#ifdef EXT_TRI
		if (extraData)
			for (int j = 0; j < 3; j++)
				tri.m_sHostData.ExtraData = extraData[indices ? indices[ti * 3 + j] : ti * 3 + j];
#endif
		triData[triIndex++] = tri;
		if (subMeshes[si] + pc <= ti)
		{
			pc += subMeshes[si];
			si++;
			if (!Les[si].isZero())
				lights.push_back(e_MeshPartLight(mats[si].Name, Les[si]));
		}
	}
	a_Out << box;
	a_Out << (unsigned int)lights.size();
	if (lights.size())
		a_Out.Write(&lights[0], lights.size() * sizeof(e_MeshPartLight));
	a_Out << numTriangles;
	a_Out.Write(triData, sizeof(e_TriangleData) * numTriangles);
	a_Out << (unsigned int)mats.size();
	a_Out.Write(&mats[0], sizeof(e_KernelMaterial) * (unsigned int)mats.size());
	ConstructBVH(vertices, indices, nVertices, numTriangles * 3, a_Out);
#ifdef EXT_TRI
	delete[] v_Normals;
	delete[] v_Tangents;
	delete[] v_BiTangents;
#endif
	delete[] triData;
}

}