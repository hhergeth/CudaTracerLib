#pragma once

#include <Math/Vector.h>
#include <Math/Spectrum.h>
#include <Math/AABB.h>
#include <Base/Buffer_device.h>
#include <Base/FixedString.h>
#include <vector>

namespace CudaTracerLib {

struct KernelMesh
{
	unsigned int m_uTriangleOffset;
	unsigned int m_uBVHNodeOffset;
	unsigned int m_uBVHTriangleOffset;
	unsigned int m_uBVHIndicesOffset;
	unsigned int m_uStdMaterialOffset;
};

struct MeshPartLight
{
	FixedString<32> MatName;
	Spectrum L;
	MeshPartLight()
	{

	}
	MeshPartLight(const std::string& name, const Spectrum& l)
		: MatName(name), L(l)
	{

	}
};

#define MESH_STATIC_TOKEN 1
#define MESH_ANIMAT_TOKEN 2

class IInStream;
class FileOutputStream;
template<typename T> class Stream;
struct TriangleData;
struct Material;
struct BVHNodeData;
struct TriIntersectorData;
struct TriIntersectorData2;
struct SceneInitData;

class Mesh
{
public:
	AABB m_sLocalBox;
	int m_uType;
public:
	StreamReference<TriangleData> m_sTriInfo;
	StreamReference<Material> m_sMatInfo;
	StreamReference<BVHNodeData> m_sNodeInfo;
	StreamReference<TriIntersectorData> m_sIntInfo;
	StreamReference<TriIntersectorData2> m_sIndicesInfo;
	std::vector<MeshPartLight> m_sAreaLights;
	FixedString<64> m_uPath;
public:
	CTL_EXPORT Mesh(const std::string& path, IInStream& a_In, Stream<TriIntersectorData>* a_Stream0,
			Stream<TriangleData>* a_Stream1, Stream<BVHNodeData>* a_Stream2, Stream<TriIntersectorData2>* a_Stream3,
			Stream<Material>* a_Stream4, Stream<char>* a_Stream5);
	CTL_EXPORT void Free(Stream<TriIntersectorData>* a_Stream0, Stream<TriangleData>* a_Stream1, Stream<BVHNodeData>* a_Stream2,
			Stream<TriIntersectorData2>* a_Stream3, Stream<Material>* a_Stream4);
	CTL_EXPORT KernelMesh getKernelData();
	unsigned int getTriangleCount()
	{
		return m_sTriInfo.getLength();
	}
	///normals, uvs, indices can be null
	///maxSmoothAngle can be used to use face normals for faces where one of the vertex normals is too different (maxSmoothAngle) from the face normal
	CTL_EXPORT static void CompileMesh(const Vec3f* vertices, unsigned int nVertices, const Vec3f* normals, const Vec2f* uvs,
			const unsigned int* indices, unsigned int nIndices, const Material& mat, const Spectrum& Le, FileOutputStream& out, bool flipNormals = false, bool faceNormals = false, float maxSmoothAngle = 0.0f);
	CTL_EXPORT static void CompileMesh(const Vec3f* vertices, unsigned int nVertices, const Vec3f* normals, const Vec2f** uvs,
			unsigned int nUV_Sets, const unsigned int* indices, unsigned int nIndices, const Material* mats,
			const Spectrum* Le, const unsigned int* subMeshes, const unsigned char* extraData, FileOutputStream& out, bool flipNormals = false, bool faceNormals = false, float maxSmoothAngle = 0.0f);
	CTL_EXPORT static SceneInitData ParseBinary(const std::string& a_InputFile);
	CTL_EXPORT static void ComputeVertexNormals(const Vec3f* V, const unsigned int* I, unsigned int vertexCount, unsigned int triCount, NormalizedT<Vec3f>* a_Normals, bool flipNormals);
};

}
