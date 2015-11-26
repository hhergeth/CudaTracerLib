#pragma once

#include <MathTypes.h>
#include "KernelDynamicScene.h"
#include "MeshCompiler.h"
#include "SceneInitData.h"
#include "ShapeSet.h"
#include <functional>

namespace CudaTracerLib {

struct e_TmpVertex;
class AnimatedMesh;
template<typename H, typename D> class BufferReference;
template<typename T> class Stream;
template<typename H, typename D> class CachedBuffer;
class SceneBVH;
struct Sensor;
struct KernelMIPMap;
class MIPMap;
class Mesh;
template<typename H, typename D> class BufferRange;

struct textureLoader;

class IFileManager
{
public:
	virtual std::string getCompiledMeshPath(const std::string& name) = 0;
	virtual std::string getTexturePath(const std::string& name) = 0;
	virtual std::string getCompiledTexturePath(const std::string& name) = 0;
};

class DynamicScene
{
	class MatStream;
private:
	std::vector<BufferReference<Node, Node>> m_sRemovedNodes;
	e_TmpVertex* m_pDeviceTmpFloats;
	e_TmpVertex* m_pHostTmpFloats;
	unsigned int m_uEnvMapIndex;
	AABB m_psSceneBoxEnvLight;
	SceneBVH* m_pBVH;
	Stream<TriangleData>* m_pTriDataStream;
	Stream<TriIntersectorData>* m_pTriIntStream;
	Stream<BVHNodeData>* m_pBVHStream;
	Stream<TriIntersectorData2>* m_pBVHIndicesStream;
	MatStream* m_pMaterialBuffer;
	CachedBuffer<MIPMap, KernelMIPMap>* m_pTextureBuffer;
	CachedBuffer<Mesh, KernelMesh>* m_pMeshBuffer;
	Stream<Node>* m_pNodeStream;
	Stream<VolumeRegion>* m_pVolumes;
	Stream<char>* m_pAnimStream;
	Stream<KernelLight>* m_pLightStream;
	MeshCompilerManager m_sCmpManager;
	Sensor* m_pCamera;
	std::function<bool(StreamReference<TriangleData>, StreamReference<TriIntersectorData>)> m_sShapeCreationClb;
	IFileManager* m_pFileManager;
protected:
	friend struct textureLoader;
	BufferReference<MIPMap, KernelMIPMap> LoadTexture(const std::string& file, bool a_MipMap);
public:
	DynamicScene(Sensor* C, SceneInitData a_Data, IFileManager* fManager);
	~DynamicScene();
	void Free();

	BufferReference<Node, Node> CreateNode(const std::string& a_MeshFile, bool force_recompile = false);
	BufferReference<Node, Node> CreateNode(const std::string& a_MeshFile, IInStream& in, bool force_recompile = false);
	BufferReference<Node, Node> CreateNode(unsigned int a_TriangleCount, unsigned int a_MaterialCount);
	void DeleteNode(BufferReference<Node, Node> ref);
	AnimatedMesh* AccessAnimatedMesh(BufferReference<Node, Node> n);
	//Creates and returns a shape structure for the submesh with material name \ref name, returning the material index optionally in \ref a_Mi
	ShapeSet CreateShape(BufferReference<Node, Node> Node, const std::string& materialIdx, unsigned int* a_Mi = 0);

	//Creates and returns an empty light
	BufferReference<KernelLight, KernelLight> createLight();
	//Creates and returns an area light in the submesh with the material nem \ref materialName
	BufferReference<KernelLight, KernelLight> createLight(BufferReference<Node, Node> Node, const std::string& materialName, Spectrum& radiance);
	//If applicable removes the light corresponding to the material with index \ref materialIdx
	void removeLight(BufferReference<Node, Node> Node, unsigned int materialIdx);
	void removeAllLights(BufferReference<Node, Node> Node);
	//Creates and returns an environment map, using \ref power as scaling factor and \ref file as texture
	BufferReference<KernelLight, KernelLight> setEnvironementMap(const Spectrum& power, const std::string& file);

	BufferReference<VolumeRegion, VolumeRegion> AddVolume(VolumeRegion& r);
	//Creates a volume with a grid size of \ref {w, h, d} and a transformation \ref worldToVol
	BufferReference<VolumeRegion, VolumeRegion> AddVolume(int w, int h, int d, const float4x4& worldToVol, const PhaseFunction& p);
	//Creates a volume with seperate grid sizes for absorption, scattering and emission and a transformation \ref worldToVol
	BufferReference<VolumeRegion, VolumeRegion> AddVolume(int wA, int hA, int dA,
		int wS, int hS, int dS,
		int wL, int hL, int dL, const float4x4& worldToVol, const PhaseFunction& p);

	void ReloadTextures();
	float4x4 GetNodeTransform(BufferReference<Node, Node> n);
	void SetNodeTransform(const float4x4& mat, BufferReference<Node, Node> n);
	void AnimateMesh(BufferReference<Node, Node> n, float t, unsigned int anim);
	//Updates the buffer contents, rebuilds the acceleration bvh and returns true when there was a change to geometry
	bool UpdateScene();
	//Instanciate the materials in \ref node so that nodes with the same mesh can have different materials
	void instanciateNodeMaterials(BufferReference<Node, Node> node);
	//Tells the acceleratio bvh that \ref node has been updated 
	void InvalidateNodesInBVH(BufferReference<Node, Node> node);
	//Tells the acceleration bvh that \ref mesh has been updated and all nodes using it will be invalidated
	void InvalidateMeshesInBVH(BufferReference<Mesh, KernelMesh> mesh);
	KernelDynamicScene getKernelSceneData(bool devicePointer = true);
	//Returns the accumulated size of all cuda allocations from buffers and textures
	size_t getCudaBufferSize();
	std::string printInfo();
	void setCamera(Sensor* C)
	{
		m_pCamera = C;
	}
	Sensor* getCamera()
	{
		return m_pCamera;
	}
	MeshCompilerManager& getMeshCompileManager()
	{
		return m_sCmpManager;
	}
	Stream<char>* getTempBuffer()
	{
		return m_pAnimStream;
	}
	SceneBVH* getBVH()
	{
		return m_pBVH;
	}
	void setShapeCreationClb(const std::function<bool(StreamReference<TriangleData>, StreamReference<TriIntersectorData>)>& clb)
	{
		m_sShapeCreationClb = clb;
	}

	BufferRange<Node, Node>& getNodes();
	BufferRange<VolumeRegion, VolumeRegion>& getVolumes();
	BufferRange<KernelLight, KernelLight>& getLights();
	BufferRange<MIPMap, KernelMIPMap>& getTextures();
	BufferRange<Mesh, KernelMesh>& getMeshes();
	BufferRange<Material, Material>& getMaterials();

	//Returns the aabb of the submesh with name \ref name, returning the material index optionally in \ref a_Mi
	AABB getAABB(BufferReference<Node, Node> Node, const std::string& name, unsigned int* a_Mi = 0);
	BufferReference<Mesh, KernelMesh> getMesh(BufferReference<Node, Node> n);
	BufferReference<Material, Material> getMaterials(BufferReference<Node, Node> n);
	BufferReference<Material, Material> getMaterial(BufferReference<Node, Node> n, const std::string& name);
	//Returns the union of all world aabbs of the nodes provided
	AABB getNodeBox(BufferReference<Node, Node> n);
	//Returns the aabb containing the scene, this will be faster than calling getNodeBox with all nodes
	AABB getSceneBox();
	unsigned int getLightCount();
	//Enumerates all lights in \ref node and calls \ref clb for each, returning the number of lights in \ref node
	size_t enumerateLights(StreamReference<Node> node, std::function<void(StreamReference<KernelLight>)> clb);
	size_t getLightCount(StreamReference<Node> node)
	{
		size_t i = 0;
		enumerateLights(node, [&](StreamReference<KernelLight>)
		{
			i++;
		});
		return i;
	}
};

}