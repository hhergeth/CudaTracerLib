#pragma once

#include <MathTypes.h>
#include "e_KernelDynamicScene.h"
#include "e_MeshCompiler.h"
#include "e_SceneInitData.h"
#include "e_ShapeSet.h"
#include <functional>

namespace CudaTracerLib {

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
template<typename H, typename D> class e_BufferRange;

struct textureLoader;

class IFileManager
{
public:
	virtual std::string getCompiledMeshPath(const std::string& name) = 0;
	virtual std::string getTexturePath(const std::string& name) = 0;
	virtual std::string getCompiledTexturePath(const std::string& name) = 0;
};

class e_DynamicScene
{
	class MatStream;
private:
	std::vector<e_BufferReference<e_Node, e_Node>> m_sRemovedNodes;
	e_TmpVertex* m_pDeviceTmpFloats;
	e_TmpVertex* m_pHostTmpFloats;
	unsigned int m_uEnvMapIndex;
	AABB m_psSceneBoxEnvLight;
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
	e_MeshCompilerManager m_sCmpManager;
	e_Sensor* m_pCamera;
	std::function<bool(e_StreamReference<e_TriangleData>, e_StreamReference<e_TriIntersectorData>)> m_sShapeCreationClb;
	IFileManager* m_pFileManager;
protected:
	friend struct textureLoader;
	e_BufferReference<e_MIPMap, e_KernelMIPMap> LoadTexture(const std::string& file, bool a_MipMap);
public:
	e_DynamicScene(e_Sensor* C, e_SceneInitData a_Data, IFileManager* fManager);
	~e_DynamicScene();
	void Free();

	e_BufferReference<e_Node, e_Node> CreateNode(const std::string& a_MeshFile, bool force_recompile = false);
	e_BufferReference<e_Node, e_Node> CreateNode(const std::string& a_MeshFile, IInStream& in, bool force_recompile = false);
	e_BufferReference<e_Node, e_Node> CreateNode(unsigned int a_TriangleCount, unsigned int a_MaterialCount);
	void DeleteNode(e_BufferReference<e_Node, e_Node> ref);
	e_AnimatedMesh* AccessAnimatedMesh(e_BufferReference<e_Node, e_Node> n);
	//Creates and returns a shape structure for the submesh with material name \ref name, returning the material index optionally in \ref a_Mi
	ShapeSet CreateShape(e_BufferReference<e_Node, e_Node> Node, const std::string& materialIdx, unsigned int* a_Mi = 0);

	//Creates and returns an empty light
	e_BufferReference<e_KernelLight, e_KernelLight> createLight();
	//Creates and returns an area light in the submesh with the material nem \ref materialName
	e_BufferReference<e_KernelLight, e_KernelLight> createLight(e_BufferReference<e_Node, e_Node> Node, const std::string& materialName, Spectrum& radiance);
	//If applicable removes the light corresponding to the material with index \ref materialIdx
	void removeLight(e_BufferReference<e_Node, e_Node> Node, unsigned int materialIdx);
	void removeAllLights(e_BufferReference<e_Node, e_Node> Node);
	//Creates and returns an environment map, using \ref power as scaling factor and \ref file as texture
	e_BufferReference<e_KernelLight, e_KernelLight> setEnvironementMap(const Spectrum& power, const std::string& file);

	e_BufferReference<e_VolumeRegion, e_VolumeRegion> AddVolume(e_VolumeRegion& r);
	//Creates a volume with a grid size of \ref {w, h, d} and a transformation \ref worldToVol
	e_BufferReference<e_VolumeRegion, e_VolumeRegion> AddVolume(int w, int h, int d, const float4x4& worldToVol, const e_PhaseFunction& p);
	//Creates a volume with seperate grid sizes for absorption, scattering and emission and a transformation \ref worldToVol
	e_BufferReference<e_VolumeRegion, e_VolumeRegion> AddVolume(int wA, int hA, int dA,
		int wS, int hS, int dS,
		int wL, int hL, int dL, const float4x4& worldToVol, const e_PhaseFunction& p);

	void ReloadTextures();
	float4x4 GetNodeTransform(e_BufferReference<e_Node, e_Node> n);
	void SetNodeTransform(const float4x4& mat, e_BufferReference<e_Node, e_Node> n);
	void AnimateMesh(e_BufferReference<e_Node, e_Node> n, float t, unsigned int anim);
	//Updates the buffer contents, rebuilds the acceleration bvh and returns true when there was a change to geometry
	bool UpdateScene();
	//Instanciate the materials in \ref node so that nodes with the same mesh can have different materials
	void instanciateNodeMaterials(e_BufferReference<e_Node, e_Node> node);
	//Tells the acceleratio bvh that \ref node has been updated 
	void InvalidateNodesInBVH(e_BufferReference<e_Node, e_Node> node);
	//Tells the acceleration bvh that \ref mesh has been updated and all nodes using it will be invalidated
	void InvalidateMeshesInBVH(e_BufferReference<e_Mesh, e_KernelMesh> mesh);
	e_KernelDynamicScene getKernelSceneData(bool devicePointer = true);
	//Returns the accumulated size of all cuda allocations from buffers and textures
	size_t getCudaBufferSize();
	std::string printInfo();
	void setCamera(e_Sensor* C)
	{
		m_pCamera = C;
	}
	e_Sensor* getCamera()
	{
		return m_pCamera;
	}
	e_MeshCompilerManager& getMeshCompileManager()
	{
		return m_sCmpManager;
	}
	e_Stream<char>* getTempBuffer()
	{
		return m_pAnimStream;
	}
	e_SceneBVH* getBVH()
	{
		return m_pBVH;
	}
	void setShapeCreationClb(const std::function<bool(e_StreamReference<e_TriangleData>, e_StreamReference<e_TriIntersectorData>)>& clb)
	{
		m_sShapeCreationClb = clb;
	}

	e_BufferRange<e_Node, e_Node>& getNodes();
	e_BufferRange<e_VolumeRegion, e_VolumeRegion>& getVolumes();
	e_BufferRange<e_KernelLight, e_KernelLight>& getLights();
	e_BufferRange<e_MIPMap, e_KernelMIPMap>& getTextures();
	e_BufferRange<e_Mesh, e_KernelMesh>& getMeshes();
	e_BufferRange<e_KernelMaterial, e_KernelMaterial>& getMateriales();

	//Returns the aabb of the submesh with name \ref name, returning the material index optionally in \ref a_Mi
	AABB getAABB(e_BufferReference<e_Node, e_Node> Node, const std::string& name, unsigned int* a_Mi = 0);
	e_BufferReference<e_Mesh, e_KernelMesh> getMesh(e_BufferReference<e_Node, e_Node> n);
	e_BufferReference<e_KernelMaterial, e_KernelMaterial> getMats(e_BufferReference<e_Node, e_Node> n);
	e_BufferReference<e_KernelMaterial, e_KernelMaterial> getMat(e_BufferReference<e_Node, e_Node> n, const std::string& name);
	//Returns the union of all world aabbs of the nodes provided
	AABB getNodeBox(e_BufferReference<e_Node, e_Node> n);
	//Returns the aabb containing the scene, this will be faster than calling getNodeBox with all nodes
	AABB getSceneBox();
	unsigned int getLightCount();
	//Enumerates all lights in \ref node and calls \ref clb for each, returning the number of lights in \ref node
	size_t enumerateLights(e_StreamReference<e_Node> node, std::function<void(e_StreamReference<e_KernelLight>)> clb);
	size_t getLightCount(e_StreamReference<e_Node> node)
	{
		size_t i = 0;
		enumerateLights(node, [&](e_StreamReference<e_KernelLight>)
		{
			i++;
		});
		return i;
	}
};

}