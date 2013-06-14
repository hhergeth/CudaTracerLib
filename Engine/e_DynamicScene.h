#pragma once

#include "..\Math\vector.h"
#include "e_Buffer.h"
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
	e_Stream<e_TriangleData>* m_pTriDataStream;
	e_Stream<e_TriIntersectorData>* m_pTriIntStream;
	e_Stream<e_BVHNodeData>* m_pBVHStream;
	e_Stream<int>* m_pBVHIndicesStream;
	e_Stream<e_KernelMaterial>* m_pMaterialBuffer;
	e_CachedBuffer<e_Texture, e_KernelTexture>* m_pTextureBuffer;
	e_CachedBuffer<e_Mesh, e_KernelMesh>* m_pMeshBuffer;
	e_Stream<e_Node>* m_pNodeStream;
	e_Stream<e_VolumeRegion>* m_pVolumes;
	e_Stream<char>* m_pAnimStream;
	e_Stream<e_KernelLight>* m_pLightStream;
	e_SceneBVH* m_pBVH;
	e_TmpVertex* m_pDeviceTmpFloats;
	e_Terrain* m_pTerrain;
public:
	e_DynamicScene(e_SceneInitData a_Data);
	~e_DynamicScene();
	void Free();
	e_StreamReference(e_Node) CreateNode(const char* a_MeshFile);
	///Do not use this! Just invalidate and update the material
	e_BufferReference<e_Texture, e_KernelTexture> LoadTexture(char* file);
	void SetNodeTransform(const float4x4& mat, e_StreamReference(e_Node) n)
	{
		m_uModified = 1;
		n->setTransform(mat);
		n.Invalidate();
		recalculateAreaLights(n);
	}
	void TransformNode(float4x4& mat, e_StreamReference(e_Node) n)
	{
		SetNodeTransform(n->getWorldMatrix() * mat, n);
	}
	void MoveNode(float3 p, e_StreamReference(e_Node) n)
	{
		TransformNode(float4x4::Translate(p), n);
	}
	template<typename MAT> void setMat(e_StreamReference(e_Node) N, unsigned int mi, MAT& m)
	{
		m_pMaterialBuffer->operator()(N->m_uMaterialOffset + mi)->SetData(m);
		m_pMaterialBuffer->operator()(N->m_uMaterialOffset + mi).Invalidate();
	}
	void AnimateMesh(e_StreamReference(e_Node) n, float t, unsigned int anim);
	void UpdateInvalidated();
	e_KernelDynamicScene getKernelSceneData();
	void UpdateMaterial(e_StreamReference(e_KernelMaterial) m);
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
	e_SceneBVH* getSceneBVH()
	{
		return m_pBVH;
	}
	unsigned int getCudaBufferSize();
	unsigned int getTriangleCount()
	{
		unsigned int r = 0;
		for(unsigned int i = 0; i < m_pNodeStream->NumUsedElements(); i++)
			r += getMesh(m_pNodeStream[0](i))->getTriangleCount();
		return r;
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
	e_StreamReference(e_VolumeRegion) AddVolume(e_VolumeRegion& r)
	{
		e_StreamReference(e_VolumeRegion) r2 = m_pVolumes->malloc(1);
		*r2.operator->() = r;
		return r2;
	}
	e_StreamReference(e_VolumeRegion) getVolumes()
	{
		return m_pVolumes->UsedElements();
	}
	AABB getAABB(e_StreamReference(e_Node) Node, const char* name, unsigned int* a_Mi = 0)
	{
		return CreateShape<128>(Node, name, a_Mi).getBox();
	}
	e_BufferReference<e_Mesh, e_KernelMesh> getMesh(e_StreamReference(e_Node) n)
	{
		return m_pMeshBuffer->operator()(n->m_uMeshIndex);
	}
	e_StreamReference(e_KernelMaterial) getMats(e_StreamReference(e_Node) n)
	{
		return m_pMaterialBuffer->operator()(n->m_uMaterialOffset, getMesh(n)->m_sMatInfo.getLength());
	}
	template<int N> ShapeSet<N> CreateShape(e_StreamReference(e_Node) Node, const char* name, unsigned int* a_Mi = 0)
	{
		e_TriIntersectorData* n[N];
		unsigned int n2[N];
		e_BufferReference<e_Mesh, e_KernelMesh> m = getMesh(Node);
		unsigned int c = 0, mi = -1;
		
		for(int j = 0; j < m->m_sMatInfo.getLength(); j++)
			if(strstr(m->m_sMatInfo(j)->Name, name))
			{
				mi = j;
				break;
			}
		if(mi == -1)
			throw 1;
		if(a_Mi)
			*a_Mi = mi;

		int i = 0, e = m->m_sIntInfo.getLength() * 4;
		while(i < e)
		{
			e_TriIntersectorData* sec = (e_TriIntersectorData*)(m->m_sIntInfo.operator()<float4>(i));
			int* ind = (int*)m->m_sIndicesInfo(i);
			if(*ind == -1)
			{
				i++;
				continue;
			}
			if(*ind < -1 || *ind >= m->m_sTriInfo.getLength())
				break;
			e_TriangleData* d = m->m_sTriInfo(*ind);
			if(d->getMatIndex(0) == mi)//do not use Node->m_uMaterialOffset, cause mi is local...
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

		ShapeSet<N> r = ShapeSet<N>(n, c, Node->getWorldMatrix());
		return r;
	}
	template<typename T> e_StreamReference(e_KernelLight) createLight(T& val)
	{
		e_StreamReference(e_KernelLight) r = m_pLightStream->malloc(1);
		r()->Set(val);
		return r();
	}
	AABB getBox(e_StreamReference(e_Node) n)
	{
		return n->getWorldBox(getMesh(n));
	}
	e_StreamReference(e_KernelLight) createLight(e_StreamReference(e_Node) Node, const char* materialName, float3& L)
	{
		unsigned int mi;
		ShapeSet<MAX_SHAPE_LENGTH> s = CreateShape<MAX_SHAPE_LENGTH>(Node, materialName, &mi);
		unsigned int* a = &m_pMaterialBuffer->operator()(Node->m_uMaterialOffset + mi)->NodeLightIndex;
		if(*a != -1)
		{
			e_StreamReference(e_KernelLight) c = m_pLightStream->operator()(Node->m_uLightIndices[*a]);
			c->Set(e_DiffuseLight(L, s));
			c.Invalidate();
			return c;
		}
		else
		{
			unsigned int b = Node->getNextFreeLightIndex();
			if(b == -1)
				throw 1;
			*a = b;
			e_StreamReference(e_KernelLight) c = m_pLightStream->malloc(1);
			Node->m_uLightIndices[b] = c.getIndex();
			c->Set(e_DiffuseLight(L, s));
			return c;
		}
	}
	void removeLight(e_StreamReference(e_Node) Node, unsigned int mi)
	{
		unsigned int a = m_pMaterialBuffer->operator()(Node->m_uMaterialOffset + mi)->NodeLightIndex;
		if(a == -1)
			return;
		unsigned int* b = Node->m_uLightIndices + a;
		m_pLightStream->dealloc(m_pLightStream->operator()(*b));
		*b = -1;
	}
	void removeAllLights(e_StreamReference(e_Node) Node)
	{
		e_BufferReference<e_Mesh, e_KernelMesh> m = getMesh(Node);
		for(int i = 0; i < m->m_sMatInfo.getLength(); i++)
			m_pMaterialBuffer->operator()(Node->m_uMaterialOffset + i)->NodeLightIndex = -1;
		int i = 0;
		while(i < MAX_AREALIGHT_NUM && Node->m_uLightIndices[i] != -1)
		{
			m_pLightStream->dealloc(m_pLightStream->operator()(Node->m_uLightIndices[i]));
			Node->m_uLightIndices[i++] = -1;
		}
	}
	void recalculateAreaLights(e_StreamReference(e_Node) Node)
	{
		int i = 0; 
		while(Node->m_uLightIndices[i] != -1)
		{
			m_pLightStream->operator()(Node->m_uLightIndices[i])->As<e_DiffuseLight>()->shapeSet.Recalculate(Node->getWorldMatrix());
			m_pLightStream->Invalidate(Node->m_uLightIndices[i]);
			i++;
		}
	}
};
