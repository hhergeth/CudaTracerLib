#include "StdAfx.h"
#include "Mesh.h"
#include <SceneTypes/Volumes.h>
#include <SceneTypes/Light.h>
#include "MeshLoader/BVHBuilderHelper.h"
#include <Base/Buffer.h>
#include <Base/FileStream.h>
#include "SceneInitData.h"
#include "TriangleData.h"
#include "Material.h"
#include "TriIntersectorData.h"

namespace CudaTracerLib {

#define NO_NODE 0x76543210
void write(int idx, const BVHNodeData* nodes, int parent, std::ofstream& f, int& leafC)
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
void printBVHData(const BVHNodeData* bvh, const std::string& path)
{
	std::ofstream file;
	file.open(path);
	file << "digraph SceneBVH {\nnode [fontname=\"Arial\"];\n";
	int leafC = 0;
	write(0, bvh, -1, file, leafC);
	file << "}";
	file.close();
}

Mesh::Mesh(const std::string& path, IInStream& a_In, Stream<TriIntersectorData>* a_Stream0, Stream<TriangleData>* a_Stream1, Stream<BVHNodeData>* a_Stream2, Stream<TriIntersectorData2>* a_Stream3, Stream<Material>* a_Stream4, Stream<char>* a_Stream5)
	: m_uPath(path)
{
	m_uType = MESH_STATIC_TOKEN;

	a_In >> m_sLocalBox;
	unsigned int numLights;
	a_In >> numLights;
	m_sAreaLights.resize(numLights);
	if (numLights)
		a_In.Read(&m_sAreaLights[0], numLights * sizeof(MeshPartLight));

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

KernelMesh Mesh::getKernelData()
{
	KernelMesh m_sData;
	m_sData.m_uBVHIndicesOffset = m_sIndicesInfo.getIndex();
	m_sData.m_uBVHNodeOffset = m_sNodeInfo.getIndex() * sizeof(BVHNodeData) / sizeof(float4);
	m_sData.m_uBVHTriangleOffset = m_sIntInfo.getIndex() * 3;
	m_sData.m_uTriangleOffset = m_sTriInfo.getIndex();
	m_sData.m_uStdMaterialOffset = m_sMatInfo.getIndex();
	return m_sData;
}

void Mesh::Free(Stream<TriIntersectorData>* a_Stream0, Stream<TriangleData>* a_Stream1, Stream<BVHNodeData>* a_Stream2, Stream<TriIntersectorData2>* a_Stream3, Stream<Material>* a_Stream4)
{
	a_Stream0->dealloc(m_sIntInfo);
	a_Stream1->dealloc(m_sTriInfo);
	a_Stream2->dealloc(m_sNodeInfo);
	a_Stream3->dealloc(m_sIndicesInfo);
	a_Stream4->dealloc(m_sMatInfo);
}

SceneInitData Mesh::ParseBinary(const std::string& a_InputFile)
{
	FileInputStream a_In(a_InputFile);
	AABB m_sLocalBox;
	a_In >> m_sLocalBox;
	unsigned int numLights;
	a_In >> numLights;
	a_In.Move(sizeof(MeshPartLight) * numLights);
#define PRINT(n, t) { a_In.Move((int)(n * sizeof(t))); Platform::OutputDebug(format("Buf : %s, length : %llu, size : %llu[MB]\n", #t, size_t(n), size_t((n) * sizeof(t) / (1024 * 1024))));}
	unsigned int m_uTriangleCount;
	a_In >> m_uTriangleCount;
	PRINT(m_uTriangleCount, TriangleData)
	unsigned int m_uMaterialCount;
	a_In >> m_uMaterialCount;
	PRINT(m_uMaterialCount, Material)
	unsigned long long m_uNodeSize;
	a_In >> m_uNodeSize;
	PRINT(m_uNodeSize, BVHNodeData)
	unsigned long long m_uIntSize;
	a_In >> m_uIntSize;
	PRINT(m_uIntSize, TriIntersectorData)
	unsigned long long m_uIndicesSize;
	a_In >> m_uIndicesSize;
	PRINT(m_uIndicesSize, TriIntersectorData2)
#undef PRINT
#undef PRINT2
		a_In.Close();
	Platform::OutputDebug(format("return CreateForSpecificMesh(%d, %d, %d, %d, 255, a_Lights);\n", m_uTriangleCount, m_uIntSize, m_uNodeSize, m_uIndicesSize));
	return SceneInitData::CreateForSpecificMesh(m_uTriangleCount, (unsigned int)m_uIntSize, (unsigned int)m_uNodeSize, (unsigned int)m_uIndicesSize, 255, 16, 16, 8);
}

void Mesh::ComputeVertexNormals(const Vec3f* V, const unsigned int* I, unsigned int vertexCount, unsigned int triCount, NormalizedT<Vec3f>* a_Normals, bool flipNormals)
{
	Vec3f* NOR = (Vec3f*)a_Normals;
	for (unsigned int i = 0; i < vertexCount; i++)
		NOR[i] = Vec3f(0.0f);
	float flip_coeff = (flipNormals ? -1.0f : 1.0f);
	for (unsigned int f = 0; f < triCount; f++)
	{
		unsigned int i1 = I ? I[f * 3 + 0] : f * 3 + 0;
		unsigned int i2 = I ? I[f * 3 + 1] : f * 3 + 1;
		unsigned int i3 = I ? I[f * 3 + 2] : f * 3 + 2;
		const Vec3f v1 = V[i1], v2 = V[i2], v3 = V[i3];

		//normalized facet normal
		Vec3f face_nor = cross(v1 - v2, v3 - v2);
		float face_area = 0.5f * face_nor.length();
		face_nor = face_nor.normalized();

		//weighting by size of triangle
		//const Vec3f normal = flip_coeff * face_area * face_nor;
		//NOR[i1] += normal;
		//NOR[i2] += normal;
		//NOR[i3] += normal;

		//tip angle weighting
		//auto nor = [&](const Vec3f& p_base, const Vec3f& p_neigh1, const Vec3f& p_neigh2) {return flip_coeff * face_nor * math::acos(dot(p_neigh1 - p_base, p_neigh2 - p_base) / (length(p_neigh1 - p_base) * length(p_neigh2 - p_base))); };
		//NOR[i1] += nor(v1, v2, v3);
		//NOR[i2] += nor(v2, v1, v3);
		//NOR[i3] += nor(v3, v1, v2);

		//sphere inscribed polytope
		auto nor = [&](const Vec3f& p_base, const Vec3f& p_neigh1, const Vec3f& p_neigh2) {return flip_coeff * cross(p_neigh1 - p_base, p_neigh2 - p_base) / (lenSqr(p_neigh1 - p_base) * lenSqr(p_neigh2 - p_base)); };
		NOR[i1] += nor(v1, v3, v2);
		NOR[i2] += nor(v2, v1, v3);
		NOR[i3] += nor(v3, v2, v1);
	}

	for (unsigned int a = 0; a < vertexCount; a++)
		a_Normals[a] = NOR[a].normalized();
}

void Mesh::CompileMesh(const Vec3f* vertices, unsigned int nVertices, const Vec3f* normals, const Vec2f* uvs, const unsigned int* indices, unsigned int nIndices, const Material& mat, const Spectrum& Le, FileOutputStream& a_Out, bool flipNormals, bool faceNormals, float maxSmoothAngle)
{
	unsigned int N = indices ? nIndices / 3 : nVertices / 3;
	CompileMesh(vertices, nVertices, normals, uvs ? &uvs : 0, uvs ? 1 : 0, indices, nIndices, &mat, Le.isZero() ? 0 : &Le, &N, 0, a_Out, flipNormals, faceNormals, maxSmoothAngle);

}

void Mesh::CompileMesh(const Vec3f* vertices, unsigned int nVertices, const Vec3f* a_normals, const Vec2f** uvs, unsigned int nUV_Sets, const unsigned int* indices, unsigned int nIndices, const Material* mats, const Spectrum* Les, const unsigned int* subMeshes, const unsigned char* extraData, FileOutputStream& a_Out, bool flipNormals, bool faceNormals, float maxSmoothAngle)
{
	std::vector<MeshPartLight> lights;
	auto add_light = [&](int submesh_index)
	{
		if (Les && !Les[submesh_index].isZero())
			lights.push_back(MeshPartLight(mats[submesh_index].Name, Les[submesh_index]));
	};

	Vec3f p[3];
	auto* n = (NormalizedT<Vec3f>*)alloca(sizeof(NormalizedT<Vec3f>) * 3);
	Vec2f t[3];
	unsigned int numTriangles = indices ? nIndices / 3 : nVertices / 3;
	TriangleData* triData = new TriangleData[numTriangles];
#ifdef EXT_TRI
	std::vector<NormalizedT<Vec3f>> comp_normals(nVertices);
	//compute the frame for the first set and hope the rest is aligned
	if(a_normals == 0 || flipNormals)
		Mesh::ComputeVertexNormals(vertices, indices, nVertices, numTriangles, &comp_normals[0], flipNormals);
#endif
	AABB box = AABB::Identity();
	unsigned int submesh_index = 0, num_prev_triangles = 0;
	for (size_t ti = 0; ti < numTriangles; ti++)
	{
		auto v_idx = [&](unsigned int j) {return indices ? indices[ti * 3 + j] : ti * 3 + j; };
		if (num_prev_triangles + subMeshes[submesh_index] <= ti)
		{
			num_prev_triangles += subMeshes[submesh_index];
			add_light(submesh_index);
			submesh_index++;
		}
		TriangleData tri;
		tri.setMatIndex(submesh_index);
		for (unsigned int uvIdx = 0; uvIdx < DMIN2(nUV_Sets, NUM_UV_SETS); uvIdx++)
		{
			for (unsigned int j = 0; j < 3; j++)
			{
				t[j] = uvs[uvIdx][v_idx(j)];
			}
			tri.setUvSetData(uvIdx, t[0], t[1], t[2]);
		}
		//copy positions first so they can be used to compute face normal
		for (unsigned int j = 0; j < 3; j++)
		{
			p[j] = vertices[v_idx(j)];
			box = box.Extend(p[j]);
		}
#ifdef EXT_TRI
		NormalizedT<Vec3f> n_face = (p[0] - p[1]).cross(p[2] - p[1]).normalized();
		if (flipNormals)
			n_face = -n_face;
#endif
		for (unsigned int j = 0; j < 3; j++)
		{
			auto l = v_idx(j);
#ifdef EXT_TRI
			n[j] = faceNormals ? n_face : (a_normals && !flipNormals ? a_normals[l].normalized() : comp_normals[l]);
#endif
		}
#ifdef EXT_TRI
		bool use_face_normal = false;
		if (!faceNormals && maxSmoothAngle != 0)
		{
			for (unsigned int j = 0; j < 3; j++)
				if (acosf(n_face.dot(n[j])) > maxSmoothAngle)
					use_face_normal |= true;
		}
		if (use_face_normal)
			n[0] = n[1] = n[2] = n_face;
#endif
		tri.setData(p[0], p[1], p[2], n[0], n[1], n[2]);

#ifdef EXT_TRI
		if (extraData)
			for (unsigned int j = 0; j < 3; j++)
				tri.m_sHostData.ExtraData = extraData[v_idx(j)];
#endif
		triData[ti] = tri;
	}
	add_light(submesh_index);
	a_Out << box;
	a_Out << (unsigned int)lights.size();
	if (lights.size())
		a_Out.Write(&lights[0], lights.size() * sizeof(MeshPartLight));
	a_Out << numTriangles;
	a_Out.Write(triData, sizeof(TriangleData) * numTriangles);
	unsigned int nMaterials = submesh_index + 1;
	a_Out << nMaterials;
	a_Out.Write(&mats[0], sizeof(Material) * nMaterials);
	ConstructBVH(vertices, indices, nVertices, numTriangles * 3, a_Out);
	delete[] triData;
}

}