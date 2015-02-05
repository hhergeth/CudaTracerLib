#pragma once

#include <MathTypes.h>
#include "e_Buffer.h"
#include "e_Node.h"
#include "e_SceneBVH.h"
#include "e_Light.h"
#include "e_TerrainHeader.h"
#include "e_Material.h"
#include "e_Volumes.h"
#include "e_KernelDynamicScene.h"
#include "e_MeshCompiler.h"

class e_Terrain;

#include "e_SceneInitData.h"
#include "e_AnimatedMesh.h"
#include "e_Sensor.h"

class e_DynamicScene
{
private:
	unsigned int m_uModified;
public:
	e_Stream<e_TriangleData>* m_pTriDataStream;
	e_Stream<e_TriIntersectorData>* m_pTriIntStream;
	e_Stream<e_BVHNodeData>* m_pBVHStream;
	e_Stream<e_TriIntersectorData2>* m_pBVHIndicesStream;
	e_Stream<e_KernelMaterial>* m_pMaterialBuffer;
	e_CachedBuffer<e_MIPMap, e_KernelMIPMap>* m_pTextureBuffer;
	e_CachedBuffer<e_Mesh, e_KernelMesh>* m_pMeshBuffer;
	e_Stream<e_Node>* m_pNodeStream;
	e_Stream<e_VolumeRegion>* m_pVolumes;
	e_Stream<char>* m_pAnimStream;
	e_Stream<e_KernelLight>* m_pLightStream;
	e_SceneBVH* m_pBVH;
	e_TmpVertex* m_pDeviceTmpFloats;
	e_Terrain* m_pTerrain;
	const char* m_pTexturePath;
	const char* m_pCompilePath;
	e_MeshCompilerManager m_sCmpManager;
	e_Sensor* m_pCamera;
	unsigned int m_uEnvMapIndex;
	bool instanciatedMaterials;
public:
	e_DynamicScene(e_Sensor* C, e_SceneInitData a_Data, const char* texPath, const char* cmpPath, const char* dataPath);
	~e_DynamicScene();
	void Free();
	e_StreamReference(e_Node) CreateNode(const char* a_MeshFile);
	e_StreamReference(e_Node) CreateNode(const char* a_MeshFile, IInStream& in);
	e_StreamReference(e_Node) CreateNode(unsigned int a_TriangleCount, unsigned int a_MaterialCount);
	void DeleteNode(e_StreamReference(e_Node) ref);
	///Do not use this! Just invalidate and update the material
	e_BufferReference<e_MIPMap, e_KernelMIPMap> LoadTexture(const char* file, bool a_MipMap);
	void UnLoadTexture(e_BufferReference<e_MIPMap, e_KernelMIPMap> ref);
	void ReloadTextures();
	float4x4 GetNodeTransform(unsigned int i)
	{
		return GetNodeTransform(getNodes()(i));
	}
	float4x4 GetNodeTransform(e_StreamReference(e_Node) n)
	{
		return *m_pBVH->m_pTransforms[0](n.getIndex());
	}
	void SetNodeTransform(const float4x4& mat, e_StreamReference(e_Node) n);
	void TransformNode(const float4x4& mat, e_StreamReference(e_Node) n)
	{
		SetNodeTransform(mat % GetNodeTransform(n), n);
	}
	void MoveNode(const Vec3f& p, e_StreamReference(e_Node) n)
	{
		TransformNode(float4x4::Translate(p), n);
	}
	template<typename MAT> void setMat(e_StreamReference(e_Node) N, unsigned int mi, MAT& m)
	{
		m_pMaterialBuffer->operator()(N->m_uMaterialOffset + mi)->bsdf.SetData(m);
		m_pMaterialBuffer->operator()(N->m_uMaterialOffset + mi).Invalidate();
	}
	void AnimateMesh(e_StreamReference(e_Node) n, float t, unsigned int anim);
	e_AnimatedMesh* AccessAnimatedMesh(e_StreamReference(e_Node) n)
	{
		e_BufferReference<e_Mesh, e_KernelMesh> m = getMesh(n);
		e_AnimatedMesh* m2 = (e_AnimatedMesh*)m.operator->();
		return m2;
	}
	bool UpdateScene();
	e_KernelDynamicScene getKernelSceneData(bool devicePointer = true)  const;
	//void UpdateMaterial(e_StreamReference(e_KernelMaterial) m);
	e_StreamReference(e_Node) getNodes()
	{
		return m_pNodeStream->UsedElements();
	}
	unsigned int getNodeCount()
	{
		return m_pNodeStream->NumUsedElements();
	}
	unsigned int getLightCount()
	{
		return m_pLightStream->NumUsedElements();
	}
	e_StreamReference(e_KernelLight) getLights()
	{
		return m_pLightStream->UsedElements();
	}
	unsigned int getMaterialCount()
	{
		return m_pMaterialBuffer->NumUsedElements();
	}
	e_StreamReference(e_KernelMaterial) getMaterials()
	{
		return m_pMaterialBuffer->UsedElements();
	}
	e_SceneBVH* getSceneBVH();
	unsigned int getCudaBufferSize();
	unsigned int getTriangleCount()
	{
		unsigned int r = 0;
		for(unsigned int i = 0; i < m_pNodeStream->NumUsedElements(); i++)
			r += getMesh(m_pNodeStream[0](i))->getTriangleCount();
		return r;
	}
	void setTerrain(e_Terrain* T);
	void printStatus(char* dest);
	void setCamera(e_Sensor* C)
	{
		m_pCamera = C;
	}
	e_Sensor* getCamera()
	{
		return m_pCamera;
	}
	e_Terrain* getTerrain();
	e_StreamReference(e_VolumeRegion) AddVolume(e_VolumeRegion& r);
	e_StreamReference(e_VolumeRegion) AddVolume(int w, int h, int d, const float4x4& worldToVol, const e_PhaseFunction& p);
	e_StreamReference(e_VolumeRegion) AddVolume(int wA, int hA, int dA,
												int wS, int hS, int dS,
												int wL, int hL, int dL, const float4x4& worldToVol, const e_PhaseFunction& p);
	e_StreamReference(e_VolumeRegion) getVolumes();
	AABB getAABB(e_StreamReference(e_Node) Node, const char* name, unsigned int* a_Mi = 0);
	e_BufferReference<e_Mesh, e_KernelMesh> getMesh(e_StreamReference(e_Node) n);
	e_StreamReference(e_KernelMaterial) getMats(e_StreamReference(e_Node) n);
	e_StreamReference(e_KernelMaterial) getMat(e_StreamReference(e_Node) n, const char* name);
	ShapeSet CreateShape(e_StreamReference(e_Node) Node, const char* name, unsigned int* a_Mi = 0);
	template<typename T> e_StreamReference(e_KernelLight) createLight(T& val)
	{
		e_StreamReference(e_KernelLight) r = m_pLightStream->malloc(1);
		r()->SetData(val);
		return r();
	}
	AABB getBox(e_StreamReference(e_Node) n);
	e_StreamReference(e_KernelLight) createLight(e_StreamReference(e_Node) Node, const char* materialName, Spectrum& L);
	void removeLight(e_StreamReference(e_Node) Node, unsigned int mi);
	void removeAllLights(e_StreamReference(e_Node) Node);
	void recalculateAreaLights(e_StreamReference(e_Node) Node);
	e_StreamReference(e_KernelLight) setEnvironementMap(const Spectrum& power, const char* file);
	void InvalidateSceneBVH()
	{
		m_uModified = 1;
	}
	unsigned int getLightCount(e_StreamReference(e_Node) n)
	{
		int i = 0;
		while (n->m_uLightIndices[i] != 0xffffffff && i < MAX_AREALIGHT_NUM)
			i++;
		return i;
	}
	e_StreamReference(e_KernelLight) getLight(e_StreamReference(e_Node) n, unsigned int i)
	{
		return m_pLightStream->operator()(n->m_uLightIndices[i], 1);
	}
};