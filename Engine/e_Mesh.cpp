#include "StdAfx.h"
#include "e_Mesh.h"
#include "SceneBuilder\TangentSpaceHelper.h"
#include "e_Volumes.h"
#include "e_Light.h"
#include "SceneBuilder\Importer.h"
#include "e_Buffer.h"
#include "../Base/FileStream.h"
#include "e_SceneInitData.h"
#include "e_TriangleData.h"
#include "e_Material.h"
#include "e_IntersectorData.h"

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

void createLeafInfo(Vec2i* data, int idx, int parent, int sibling, e_BVHNodeData* data2)
{
	if (idx < 0)
	{
		data[~idx] = Vec2i(parent, sibling);
	}
	else
	{
		e_BVHNodeData& n = data2[idx / 4];
		Vec2i c = n.getChildren();
		if (c.x != 0x76543210)
			createLeafInfo(data, c.x, idx, c.y, data2);
		if (c.y != 0x76543210)
			createLeafInfo(data, c.y, idx, c.x, data2);
	}
}

e_Mesh::e_Mesh(const std::string& path, IInStream& a_In, e_Stream<e_TriIntersectorData>* a_Stream0, e_Stream<e_TriangleData>* a_Stream1, e_Stream<e_BVHNodeData>* a_Stream2, e_Stream<e_TriIntersectorData2>* a_Stream3, e_Stream<e_KernelMaterial>* a_Stream4, e_Stream<char>* a_Stream5)
	: m_uPath(path)
{
	m_uType = MESH_STATIC_TOKEN;
	int abc = sizeof(e_TriangleData), abc2 = sizeof(m_sLights);

	a_In >> m_sLocalBox;
	a_In.Read(m_sLights, sizeof(m_sLights));
	a_In >> m_uUsedLights;

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

	m_sLeafInfo = a_Stream5->malloc(RND_16(m_uIndicesSize * sizeof(Vec2i)));
	createLeafInfo((Vec2i*)m_sLeafInfo.operator char *(), 0, -1, -1, m_sNodeInfo());
	m_sLeafInfo.Invalidate();
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
	m_sData.m_uLeafInfoOffset = m_sLeafInfo.getIndex();
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
	InputStream a_In(a_InputFile);
	AABB m_sLocalBox;
	a_In >> m_sLocalBox;
	a_In.Move(sizeof(e_MeshPartLight) * MAX_AREALIGHT_NUM + 8);
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