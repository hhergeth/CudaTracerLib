#pragma once

#include <MathTypes.h>
#include "e_Buffer_device.h"
#include <Base/FixedString.h>
#include <vector>

struct e_KernelMesh
{
	unsigned int m_uTriangleOffset;
	unsigned int m_uBVHNodeOffset;
	unsigned int m_uBVHTriangleOffset;
	unsigned int m_uBVHIndicesOffset;
	unsigned int m_uStdMaterialOffset;
};

struct e_MeshPartLight
{
	FixedString<32> MatName;
	Spectrum L;
	e_MeshPartLight()
	{

	}
	e_MeshPartLight(const std::string& name, const Spectrum& l)
		: MatName(name), L(l)
	{

	}
};

#define MESH_STATIC_TOKEN 1
#define MESH_ANIMAT_TOKEN 2

class IInStream;
class FileOutputStream;
template<typename T> class e_Stream;
struct e_TriangleData;
struct e_KernelMaterial;
struct e_BVHNodeData;
struct e_TriIntersectorData;
struct e_TriIntersectorData2;
struct e_SceneInitData;

class e_Mesh
{
public:
	AABB m_sLocalBox;
	int m_uType;
public:
	e_StreamReference<e_TriangleData> m_sTriInfo;
	e_StreamReference<e_KernelMaterial> m_sMatInfo;
	e_StreamReference<e_BVHNodeData> m_sNodeInfo;
	e_StreamReference<e_TriIntersectorData> m_sIntInfo;
	e_StreamReference<e_TriIntersectorData2> m_sIndicesInfo;
	std::vector<e_MeshPartLight> m_sAreaLights;
	FixedString<64> m_uPath;
public:
	e_Mesh(const std::string& path, IInStream& a_In, e_Stream<e_TriIntersectorData>* a_Stream0, e_Stream<e_TriangleData>* a_Stream1, e_Stream<e_BVHNodeData>* a_Stream2, e_Stream<e_TriIntersectorData2>* a_Stream3, e_Stream<e_KernelMaterial>* a_Stream4, e_Stream<char>* a_Stream5);
	void Free(e_Stream<e_TriIntersectorData>* a_Stream0, e_Stream<e_TriangleData>* a_Stream1, e_Stream<e_BVHNodeData>* a_Stream2, e_Stream<e_TriIntersectorData2>* a_Stream3, e_Stream<e_KernelMaterial>* a_Stream4);
	e_KernelMesh getKernelData();
	unsigned int getTriangleCount()
	{
		return m_sTriInfo.getLength();
	}
	static void CompileMesh(const Vec3f* vertices, unsigned int nVertices, const Vec2f* uvs, const unsigned int* indices, unsigned int nIndices, const e_KernelMaterial& mat, const Spectrum& Le, FileOutputStream& out);
	static void CompileMesh(const Vec3f* vertices, unsigned int nVertices, const Vec2f** uvs, unsigned int nUV_Sets, const unsigned int* indices, unsigned int nIndices, const std::vector<e_KernelMaterial>& mats, const std::vector<Spectrum>& Le, const std::vector<unsigned int>& subMeshes, const unsigned char* extraData, FileOutputStream& out);
	static e_SceneInitData ParseBinary(const std::string& a_InputFile);
};
