#pragma once

#include "..\Math\vector.h"
#include "e_DataStream.h"
#include "e_HostDeviceBuffer.h"
#include "e_Node.h"
#include "e_SceneBVH.h"
#include "e_Light.h"
#include "e_TerrainHeader.h"
#include "Engine/e_KernelMaterial.h"
#include "e_Volumes.h"

class e_Terrain;

struct e_KernelDynamicScene
{
	e_KernelDataStream<e_TriangleData> m_sTriData;
	e_KernelDataStream<e_TriIntersectorData> m_sBVHIntData;
	e_KernelDataStream<e_BVHNodeData> m_sBVHNodeData;
	e_KernelDataStream<int> m_sBVHIndexData;
	e_KernelDataStream<e_KernelMaterial> m_sMatData;
	e_KernelHostDeviceBuffer<e_KernelTexture> m_sTexData;
	e_KernelHostDeviceBuffer<e_KernelMesh> m_sMeshData;
	e_KernelDataStream<e_Node> m_sNodeData;
	e_KernelHostDeviceBuffer<e_KernelLight> m_sLightData;
	e_KernelDataStream<char> m_sAnimData;
	e_KernelSceneBVH m_sSceneBVH;
	e_KernelTerrainData m_sTerrain;
	e_KernelAggregateVolume m_sVolume;
};

#include "e_SceneInitData.h"
#include "e_AnimatedMesh.h"

class e_DynamicScene
{
private:
	unsigned int m_uModified;
public:
	e_DataStream<e_TriangleData>* m_pTriDataStream;
	e_DataStream<e_TriIntersectorData>* m_pTriIntStream;
	e_DataStream<e_BVHNodeData>* m_pBVHStream;
	e_DataStream<int>* m_pBVHIndicesStream;
	e_DataStream<e_KernelMaterial>* m_pMaterialBuffer;
	e_CachedHostDeviceBuffer<e_Texture, e_KernelTexture>* m_pTextureBuffer;
	e_CachedHostDeviceBuffer<e_Mesh, e_KernelMesh>* m_pMeshBuffer;
	e_DataStream<e_Node>* m_pNodeStream;
	e_DataStream<e_VolumeRegion>* m_pVolumes;
	e_DataStream<char>* m_pAnimStream;
	e_HostDeviceBuffer<e_Light, e_KernelLight>* m_pLightStream;
	e_SceneBVH* m_pBVH;
	e_TmpVertex* m_pDeviceTmpFloats;
	e_Terrain* m_pTerrain;
public:
	e_DynamicScene(e_SceneInitData a_Data);
	e_DynamicScene(InputStream& a_In);
	~e_DynamicScene();
	void Free();
	void Serialize(OutputStream& a_Out);
	e_Node* CreateNode(const char* a_MeshFile);
	e_HostDeviceBufferReference<e_Texture, e_KernelTexture> LoadTexture(char* file);
	void SetNodeTransform(float4x4& mat, e_Node* a_Node)
	{
		m_uModified = 1;
		a_Node->setTransform(mat);
		m_pNodeStream->Invalidate(DataStreamRefresh_Buffered, m_pNodeStream[0](a_Node));
	}
	void TransformNode(float4x4& mat, e_Node* a_Node)
	{
		m_uModified = 1;
		a_Node->setTransform(a_Node->getWorldMatrix() * mat);
		m_pNodeStream->Invalidate(DataStreamRefresh_Buffered, m_pNodeStream[0](a_Node));
	}
	void MoveNode(float3 p, e_Node* a_Node)
	{
		TransformNode(float4x4::Translate(p), a_Node);
	}
	void AnimateMesh(e_Node* a_Node, float t, unsigned int anim);
	void UpdateInvalidated();
	e_KernelDynamicScene getKernelSceneData();
	void UpdateMaterial(e_KernelMaterial* m);
	void UpdateMaterial(e_DataStreamReference<e_KernelMaterial>& m);
	e_Node* getNodes(unsigned int i = 0)
	{
		return m_pNodeStream->getHost(i);
	}
	unsigned int getNodeCount()
	{
		return m_pNodeStream->NumUsedElements();
	}
	unsigned int getLightCount()
	{
		return m_pLightStream->UsedElements();
	}
	e_Light* getLights(unsigned int i = 0)
	{
		return m_pLightStream->operator()(i);
	}
	unsigned int getMaterialCount()
	{
		return m_pMaterialBuffer->NumUsedElements();
	}
	e_DataStreamReference<e_KernelMaterial> getMaterials(unsigned int i = 0)
	{
		return m_pMaterialBuffer->operator()(m_pMaterialBuffer->operator()(i));
	}
	e_SceneBVH* getSceneBVH()
	{
		return m_pBVH;
	}
	unsigned int getCudaBufferSize();
	unsigned int getTriangleCount()
	{
		unsigned int r = 0;
		for(unsigned int i = 0; i < m_pNodeStream->NumUsedElements(); i++)
			r += m_pNodeStream[0](i)->m_pMesh->getTriangleCount();
		return r;
	}
	e_DirectionalLight* addDirectionalLight(AABB& box, float3 dir, float3 col)
	{
		e_HostDeviceBufferReference<e_Light, e_KernelLight> r = m_pLightStream->malloc(1);
		new(r(0)) e_DirectionalLight(box, dir, col);
		m_pLightStream->Invalidate(r);
		return (e_DirectionalLight*)r(0);
	}
	e_SphereLight* addSphereLight(float3& v, float rad, float3 col)
	{
		e_HostDeviceBufferReference<e_Light, e_KernelLight> r = m_pLightStream->malloc(1);
		m_pLightStream->Invalidate(r);
		new(r(0)) e_SphereLight(v, rad, col);
		return (e_SphereLight*)r(0);
	}
	e_DirectedLight* addDirectedLight(e_Node* N, AABB& src, float3 col)
	{
		e_HostDeviceBufferReference<e_Light, e_KernelLight> r = m_pLightStream->malloc(1);
		new (r(0)) e_DirectedLight(N, src.minV, src.maxV - src.minV, col);
		m_pLightStream->Invalidate(r);
		return (e_DirectedLight*)r(0);
	}
	e_DirectedLight* addDirectedLight(AABB& dest, AABB& src, float3 col)
	{
		e_HostDeviceBufferReference<e_Light, e_KernelLight> r = m_pLightStream->malloc(1);
		new (r(0)) e_DirectedLight(dest, src.minV, src.maxV - src.minV, col);
		m_pLightStream->Invalidate(r);
		return (e_DirectedLight*)r(0);
	}
	void setTerrain(e_Terrain* T);
	void printStatus(char* dest)
	{
		sprintf(dest, "Triangle intersectors : %d/%d\nBVH nodes : %d/%d\nBVH indices : %d/%d\nMaterials : %d/%d\nTextures : %d/%d\nMeshes : %d/%d\nNodes : %d/%d\nLights : %d/%d\n"
			, m_pTriIntStream->UsedElements(), m_pTriIntStream->getLength(), m_pBVHStream->UsedElements(), m_pBVHStream->getLength(), m_pBVHIndicesStream->UsedElements(), m_pBVHIndicesStream->getLength()
			, m_pMaterialBuffer->UsedElements(), m_pMaterialBuffer->getLength(), m_pTextureBuffer->UsedElements(), m_pTextureBuffer->getLength(), m_pMeshBuffer->UsedElements(), m_pMeshBuffer->getLength()
			, m_pNodeStream->UsedElements(), m_pNodeStream->getLength(), m_pLightStream->UsedElements(), m_pLightStream->getLength());
	}
	e_Terrain* getTerrain()
	{
		return m_pTerrain;
	}
	void AddVolume(e_VolumeRegion& r)
	{
		e_DataStreamReference<e_VolumeRegion> r2 = m_pVolumes->malloc(1);
		*r2() = r;
	}
	e_DataStreamReference<e_VolumeRegion> getVolumes()
	{
		return m_pVolumes->UsedElements();
	}
	void InstanciateMaterials(e_Node* N)
	{
		if(N->usesInstanciatedMaterials())
			return;
		e_DataStreamReference<e_KernelMaterial> mats = m_pMaterialBuffer->malloc(N->m_pMesh->m_sMatInfo);
		for(int i = 0; i < mats.getLength(); i++)
			*mats(i) = *N->m_pMesh->m_sMatInfo(i);
		mats.Invalidate();
	}
	e_Light* createLight(e_Node* N, char* name, const float3& col);
	e_Light* createLight(e_Node* N, const float3& col, char* sourceName, char* destName);
};
