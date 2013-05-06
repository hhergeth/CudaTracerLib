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
#include "e_KernelDynamicScene.h"

class e_Terrain;

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
	e_DataStream<e_KernelLight>* m_pLightStream;
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
	void SetNodeTransform(const float4x4& mat, e_Node* a_Node)
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
		return m_pLightStream->UsedElements().getLength();
	}
	e_KernelLight* getLights(unsigned int i = 0)
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
	/*e_DirectionalLight* addDirectionalLight(AABB& box, float3 dir, float3 col)
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
	}*/
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
	AABB getAABB(e_Node* N, char* name, unsigned int* a_Mi = 0);
	template<int N> ShapeSet<N> CreateShape(e_Node* Node, char* name)
	{
		e_TriIntersectorData* n[N];
		unsigned int n2[N];
		unsigned int c = 0, mi = -1;
		
		for(int j = 0; j < Node->m_pMesh->m_sMatInfo.getLength(); j++)
			if(strstr(Node->m_pMesh->m_sMatInfo(j)->Name, name))
			{
				mi = j;
				break;
			}
		if(mi == -1)
			throw 1;

		int i = 0, e = Node->m_pMesh->m_sIntInfo.getLength() * 4;
		while(i < e)
		{
			e_TriIntersectorData* sec = (e_TriIntersectorData*)((float4*)Node->m_pMesh->m_sIntInfo.operator()(0) + i);
			int* ind = Node->m_pMesh->m_sIndicesInfo(i);
			if(*ind == -1)
			{
				i++;
				continue;
			}
			if(*ind < -1 || *ind >= Node->m_pMesh->m_sTriInfo.getLength())
				break;
			e_TriangleData* d = Node->m_pMesh->m_sTriInfo(*ind);
			if(d->getMatIndex(Node->m_uMaterialOffset) == mi)
			{
				int k = 0;
				for(; k < c; k++)
					if(n2[k] == *ind)
						break;
				if(k == c)
				{
					n[c] = sec;
					n2[c++] = *ind;
				}
			}
			i += 3;
		}

		ShapeSet<N> r = ShapeSet<N>(n, m_pTriIntStream->getHost(0), n2, c);
		return r;
	}
	template<typename T> e_KernelLight* creatLight(T& val)
	{
		e_DataStreamReference<e_KernelLight> r = m_pLightStream->malloc(1);
		r()->Set(val);
		return r();
	}
	/*e_Light* createLight(e_Node* N, char* name, const float3& col);
	e_Light* createLight(e_Node* N, const float3& col, char* sourceName, char* destName);*/
};
