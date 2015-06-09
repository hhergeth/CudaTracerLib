#pragma once

#include <MathTypes.h>
#include "e_KernelDynamicScene.h"
#include "e_MeshCompiler.h"
#include "e_SceneInitData.h"
#include "e_ShapeSet.h"

struct e_TmpVertex;
class e_AnimatedMesh;
template<typename H, typename D> class e_BufferReference;
template<typename T> class e_Stream;
template<typename H, typename D> class e_CachedBuffer;
class e_SceneBVH;
struct e_Sensor;
struct e_KernelMIPMap;
class e_MIPMap;
class e_Mesh;
#ifndef e_StreamReference
#define e_StreamReference(T) e_BufferReference<T, T>
#endif

struct textureLoader;

class e_DynamicScene
{
	class MatStream;
private:
	std::vector<e_StreamReference(e_Node)> m_sRemovedNodes;
	e_TmpVertex* m_pDeviceTmpFloats;
	unsigned int m_uEnvMapIndex;
public:
	e_SceneBVH* m_pBVH;
	e_Stream<e_TriangleData>* m_pTriDataStream;
	e_Stream<e_TriIntersectorData>* m_pTriIntStream;
	e_Stream<e_BVHNodeData>* m_pBVHStream;
	e_Stream<e_TriIntersectorData2>* m_pBVHIndicesStream;
	MatStream* m_pMaterialBuffer;
	e_CachedBuffer<e_MIPMap, e_KernelMIPMap>* m_pTextureBuffer;
	e_CachedBuffer<e_Mesh, e_KernelMesh>* m_pMeshBuffer;
	e_Stream<e_Node>* m_pNodeStream;
	e_Stream<e_VolumeRegion>* m_pVolumes;
	e_Stream<char>* m_pAnimStream;
	e_Stream<e_KernelLight>* m_pLightStream;
	std::string m_pTexturePath;
	std::string m_pCompilePath;
	e_MeshCompilerManager m_sCmpManager;
	e_Sensor* m_pCamera;
protected:
	friend struct textureLoader;
	e_BufferReference<e_MIPMap, e_KernelMIPMap> LoadTexture(const std::string& file, bool a_MipMap);
public:
	e_DynamicScene(e_Sensor* C, e_SceneInitData a_Data, const std::string& texPath, const std::string& cmpPath, const std::string& dataPath);
	~e_DynamicScene();
	void Free();
	e_StreamReference(e_Node) CreateNode(const std::string& a_MeshFile, bool force_recompile = false);
	e_StreamReference(e_Node) CreateNode(const std::string& a_MeshFile, IInStream& in, bool force_recompile = false);
	e_StreamReference(e_Node) CreateNode(unsigned int a_TriangleCount, unsigned int a_MaterialCount);
	void DeleteNode(e_StreamReference(e_Node) ref);
	void ReloadTextures();
	float4x4 GetNodeTransform(e_StreamReference(e_Node) n);
	void SetNodeTransform(const float4x4& mat, e_StreamReference(e_Node) n);
	void AnimateMesh(e_StreamReference(e_Node) n, float t, unsigned int anim);
	bool UpdateScene();
	e_KernelDynamicScene getKernelSceneData(bool devicePointer = true);
	//void UpdateMaterial(e_StreamReference(e_KernelMaterial) m);
	e_StreamReference(e_Node) getNodes();
	unsigned int getNodeCount();
	unsigned int getLightCount();
	e_StreamReference(e_KernelLight) getLights();
	unsigned int getMaterialCount();
	e_StreamReference(e_KernelMaterial) getMaterials();
	e_AnimatedMesh* AccessAnimatedMesh(e_StreamReference(e_Node) n);
	unsigned int getCudaBufferSize();
	unsigned int getTriangleCount();
	std::string printStatus();
	void setCamera(e_Sensor* C)
	{
		m_pCamera = C;
	}
	e_Sensor* getCamera()
	{
		return m_pCamera;
	}
	e_StreamReference(e_VolumeRegion) AddVolume(e_VolumeRegion& r);
	e_StreamReference(e_VolumeRegion) AddVolume(int w, int h, int d, const float4x4& worldToVol, const e_PhaseFunction& p);
	e_StreamReference(e_VolumeRegion) AddVolume(int wA, int hA, int dA,
												int wS, int hS, int dS,
												int wL, int hL, int dL, const float4x4& worldToVol, const e_PhaseFunction& p);
	e_StreamReference(e_VolumeRegion) getVolumes();
	AABB getAABB(e_StreamReference(e_Node) Node, const std::string& name, unsigned int* a_Mi = 0);
	e_BufferReference<e_Mesh, e_KernelMesh> getMesh(e_StreamReference(e_Node) n);
	e_StreamReference(e_KernelMaterial) getMats(e_StreamReference(e_Node) n);
	e_StreamReference(e_KernelMaterial) getMat(e_StreamReference(e_Node) n, const std::string& name);
	ShapeSet CreateShape(e_StreamReference(e_Node) Node, const std::string& name, unsigned int* a_Mi = 0);
	AABB getBox(e_StreamReference(e_Node) n);
	e_StreamReference(e_KernelLight) createLight(e_StreamReference(e_Node) Node, const std::string& materialName, Spectrum& L);
	void removeLight(e_StreamReference(e_Node) Node, unsigned int mi);
	void removeAllLights(e_StreamReference(e_Node) Node);
	void recalculateAreaLights(e_StreamReference(e_Node) Node);
	e_StreamReference(e_KernelLight) setEnvironementMap(const Spectrum& power, const std::string& file);
	unsigned int getLightCount(e_StreamReference(e_Node) n);
	e_StreamReference(e_KernelLight) getLight(e_StreamReference(e_Node) n, unsigned int i);
	void instanciateNodeMaterials(e_StreamReference(e_Node) n);
	e_Stream<e_KernelMaterial>* getMatBuffer();
	void InvalidateNodesInBVH(e_StreamReference(e_Node) n);
	void InvalidateMeshesInBVH(e_BufferReference<e_Mesh, e_KernelMesh> m);
	void BuildFlatMeshBVH(e_BufferReference<e_Mesh, e_KernelMesh> m, const e_BVHNodeData* bvh, unsigned int bvhLength,
		const e_TriIntersectorData* int1, unsigned int int1Legth, const e_TriIntersectorData2* int2, unsigned int int2Legth);
};