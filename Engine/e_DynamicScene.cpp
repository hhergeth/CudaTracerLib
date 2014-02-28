#include "StdAfx.h"
#include "e_DynamicScene.h"
#include <iostream>
#include <algorithm>
#include <string>
#include "e_Terrain.h"
#include "..\Base\StringUtils.h"

struct textureLoader
{
	e_DynamicScene* S;
	textureLoader(e_DynamicScene* A)
	{
		S = A;
	}
	e_BufferReference<e_MIPMap, e_KernelMIPMap> operator()(char* file, bool a_MipMap)
	{
		return S->LoadTexture(file, a_MipMap);
	};
};

struct matUpdater
{
	e_DynamicScene* S;
	matUpdater(e_DynamicScene* A)
	{
		S = A;
	}
	void operator()(e_StreamReference(e_KernelMaterial) m)
	{
		m->LoadTextures(textureLoader(S));
	}
};

e_SceneInitData e_SceneInitData::CreateFor_S_SanMiguel(unsigned int a_SceneNodes, unsigned int a_Lights)
{
	int i = 4;
	e_SceneInitData r = CreateForSpecificMesh(1200000*i, 1200000*i, 1200000*i, 1200000*i, 4096 * 5, a_Lights, a_SceneNodes, a_SceneNodes / 2);
	//e_SceneInitData r = CreateForSpecificMesh(7880512, 9359209, 2341126, 28077626, 4096 * 5, a_Lights, a_SceneNodes);//san miguel
	//e_SceneInitData r = CreateForSpecificMesh(1,1,1,1,1,1,1);
	return CreateForSpecificMesh(100000, 100000, 100000, 1500000, 255, a_Lights, a_SceneNodes, a_SceneNodes / 2);
	//r.m_uSizeAnimStream = 16 * 1024 * 1024;
	r.m_uSizeAnimStream = 1;
	return r;
}

template<typename T> e_Stream<T>* LL(int i)
{
	e_Stream<T>* r = new e_Stream<T>(i);
	//OutputDebugString(format("%d [MB]\n", r->getSizeInBytes() / (1024 * 1024)).c_str());
	return r;
}

e_DynamicScene::e_DynamicScene(e_Sensor* C, e_SceneInitData a_Data, const char* texPath, const char* cmpPath)
	: m_uEnvMapIndex(0xffffffff), m_pCamera(C)
{
	instanciatedMaterials = false;
	m_pCompilePath = cmpPath;
	m_pTexturePath = texPath;
	int nodeC = 1 << 16, tCount = 1 << 16;
	m_uModified = 1;
	m_pAnimStream = LL<char>(a_Data.m_uSizeAnimStream + (a_Data.m_bSupportEnvironmentMap ? sizeof(Distribution2D<4096, 4096>) : 0));
	m_pTriDataStream = LL<e_TriangleData>(a_Data.m_uNumTriangles);
	m_pTriIntStream = LL<e_TriIntersectorData>(a_Data.m_uNumInt);
	m_pBVHStream = LL<e_BVHNodeData>(a_Data.m_uNumBvhNodes);
	m_pBVHIndicesStream = LL<e_TriIntersectorData2>(a_Data.m_uNumBvhIndices);
	m_pMaterialBuffer = LL<e_KernelMaterial>(a_Data.m_uNumMaterials);
	m_pMeshBuffer = new e_CachedBuffer<e_Mesh, e_KernelMesh>(a_Data.m_uNumMeshes, sizeof(e_AnimatedMesh));
	m_pNodeStream = LL<e_Node>(a_Data.m_uNumNodes);
	m_pTextureBuffer = new e_CachedBuffer<e_MIPMap, e_KernelMIPMap>(a_Data.m_uNumTextures);
	m_pLightStream = LL<e_KernelLight>(a_Data.m_uNumLights);
	m_pVolumes = LL<e_VolumeRegion>(128);
	m_pBVH = new e_SceneBVH(a_Data.m_uNumNodes);
	if(a_Data.m_uSizeAnimStream > 1024)
		CUDA_MALLOC(&m_pDeviceTmpFloats, sizeof(e_TmpVertex) * (1 << 16));
	m_pTerrain = new e_Terrain(1, make_float2(0,0), make_float2(0,0));
	unsigned int a = this->getCudaBufferSize() / (1024 * 1024);
	//if(a > 900 * 1024 * 1024)
	//	throw 1;
}

void e_DynamicScene::Free()
{//x->Free(); 
#define DEALLOC(x) delete x;
	DEALLOC(m_pTriDataStream)
	DEALLOC(m_pTriIntStream)
	DEALLOC(m_pBVHStream)
	DEALLOC(m_pBVHIndicesStream)
	DEALLOC(m_pMaterialBuffer)
	DEALLOC(m_pTextureBuffer)
	DEALLOC(m_pMeshBuffer)
	DEALLOC(m_pNodeStream)
	DEALLOC(m_pAnimStream)
	DEALLOC(m_pLightStream)
	DEALLOC(m_pVolumes)
	CUDA_FREE(m_pDeviceTmpFloats);
#undef DEALLOC
	delete m_pBVH;
	delete m_pTerrain;
}

e_DynamicScene::~e_DynamicScene()
{
	Free();
}

e_StreamReference(e_Node) e_DynamicScene::CreateNode(const char* a_MeshFile2)
{
	m_uModified = 1;
	std::string strA = toLower(a_MeshFile2);
	bool load;
	e_BufferReference<e_Mesh, e_KernelMesh> M = m_pMeshBuffer->LoadCached(strA.c_str(), &load);
	if(load)
	{
		std::string xmshPath;
		if(strA.find(".xmsh") == std::string::npos)
		{
			std::string t0 = std::string(m_pCompilePath) + getFileName(strA), cmpPath = t0.substr(0, t0.rfind('.')) + std::string(".xmsh");
			xmshPath = cmpPath;
			CreateDirectoryRecursively(getDirName(cmpPath).c_str());
			unsigned long long objStamp = GetTimeStamp(a_MeshFile2);
			unsigned long long xmshStamp = GetTimeStamp(cmpPath.c_str());
			unsigned long long si = GetFileSize(cmpPath.c_str());
			if(si <= 4 || si == -1 || objStamp != xmshStamp)
			{
				std::cout << "Started compiling mesh : " << a_MeshFile2 << "\n";
				OutputStream a_Out(cmpPath.c_str());
				e_MeshCompileType t;
				m_sCmpManager.Compile(a_MeshFile2, a_Out, &t);
				a_Out.Close();
				SetTimeStamp(cmpPath.c_str(), objStamp);
			}
		}
		else
		{
			xmshPath = strA;
		}
		InputStream I(xmshPath.c_str());
		unsigned int t;
		I >> t;
		if(t == (unsigned int)e_MeshCompileType::Static)
			new(M(0)) e_Mesh(I, m_pTriIntStream, m_pTriDataStream, m_pBVHStream, m_pBVHIndicesStream, m_pMaterialBuffer);
		else new(M(0)) e_AnimatedMesh(I, m_pTriIntStream, m_pTriDataStream, m_pBVHStream, m_pBVHIndicesStream, m_pMaterialBuffer, m_pAnimStream);
		I.Close();
		m_pMeshBuffer->Invalidate(M);
		M->m_sMatInfo.Invalidate();
	}
	else if(M->m_uType == MESH_ANIMAT_TOKEN)
	{
		e_BufferReference<e_Mesh, e_KernelMesh> oldM = M;
		M = m_pMeshBuffer->malloc(1);
		((e_AnimatedMesh*)oldM.operator->())->CreateNewMesh((e_AnimatedMesh*)M.operator->(), m_pTriIntStream, m_pTriDataStream, m_pBVHStream, m_pBVHIndicesStream, m_pMaterialBuffer, m_pAnimStream);
		m_pMeshBuffer->Invalidate(M);
	}
	e_StreamReference(e_Node) N = m_pNodeStream->malloc(1);
	e_StreamReference(e_KernelMaterial) m2 = M->m_sMatInfo;
	if(instanciatedMaterials && m_pMaterialBuffer->NumUsedElements() + M->m_sMatInfo.getLength() < m_pMaterialBuffer->getLength() - 1)
		m2 = m_pMaterialBuffer->malloc(M->m_sMatInfo);
	m2.Invalidate();
	new(N.operator->()) e_Node(M.getIndex(), M.operator->(), strA.c_str(), m2);
	unsigned int li[MAX_AREALIGHT_NUM];
	Platform::SetMemory(li, sizeof(li));
	for(unsigned int i = 0; i < M->m_uUsedLights; i++)
	{
		ShapeSet s = CreateShape(N, M->m_sLights[i].MatName);
		li[i] = createLight(e_DiffuseLight(M->m_sLights[i].L, s)).getIndex();
	}
	N->setLightData(li, M->m_uUsedLights);
	N.Invalidate();
	return N;
}

e_StreamReference(e_Node) e_DynamicScene::CreateNode(unsigned int a_TriangleCount, unsigned int a_MaterialCount)
{
	e_StreamReference(e_Node) N = m_pNodeStream->malloc(1);
	e_StreamReference(e_KernelMaterial) m2 = m_pMaterialBuffer->malloc(a_MaterialCount);
	e_BufferReference<e_Mesh, e_KernelMesh> M = m_pMeshBuffer->malloc(1);
	M->m_sIndicesInfo = m_pBVHIndicesStream->malloc(1);
	M->m_sIntInfo = m_pTriIntStream->malloc(1);
	M->m_sMatInfo = m2;
	M->m_sNodeInfo = m_pBVHStream->malloc(1);
	M->m_sTriInfo = m_pTriDataStream->malloc(a_TriangleCount);
	M->m_uType = MESH_STATIC_TOKEN;
	M->m_uUsedLights = 0;
	N->setLightData(0, 0);
	new(N.operator->()) e_Node(M.getIndex(), M.operator->(), "auto generated node", m2);
	N.Invalidate();
	return N;
}

//free material -> free textures, do not load textures twice!
void e_DynamicScene::DeleteNode(e_StreamReference(e_Node) ref)
{
	if(m_pMeshBuffer->Release(getMesh(ref)))
	{
		getMesh(ref)->Free(m_pTriIntStream, m_pTriDataStream, m_pBVHStream, m_pBVHIndicesStream, m_pMaterialBuffer);
	}
	m_pNodeStream->dealloc(ref);
}

e_BufferReference<e_MIPMap, e_KernelMIPMap> e_DynamicScene::LoadTexture(const char* file, bool a_MipMap)
{
	std::string a = GetFileSize(file) != -1 ? std::string(file) : std::string(m_pTexturePath) + std::string(file);
	if(a[a.size() - 1] == '\n')
		a = a.substr(0, a.size() - 1);
	bool load;
	e_BufferReference<e_MIPMap, e_KernelMIPMap> T = m_pTextureBuffer->LoadCached(a.c_str(), &load);
	if(load)
	{
		std::string a2 = std::string(m_pCompilePath) + "Images\\" + getFileName(a), b = a2.substr(0, a2.find('.')) + ".xtex";
		CreateDirectoryRecursively(getDirName(b).c_str());
		if((int)GetFileSize(b.c_str()) <= 0)
		{
			OutputStream a_Out(b.c_str());
			e_MIPMap::CompileToBinary(a.c_str(), a_Out, a_MipMap);
			a_Out.Close();
		}
		InputStream I(b.c_str());
		new(T) e_MIPMap(I);
		I.Close();
		T->CreateKernelTexture();
		T.Invalidate();
	}
	if(!T->getKernelData().m_pDeviceData)
		throw 1;
	return T;
}

void e_DynamicScene::UnLoadTexture(e_BufferReference<e_MIPMap, e_KernelMIPMap> ref)
{
	if(m_pTextureBuffer->Release(ref))
	{
		ref->Free();
	}
}

void e_DynamicScene::SetNodeTransform(const float4x4& mat, e_StreamReference(e_Node) n)
{
	m_uModified = 1;
	for(unsigned int i = 0; i < n.getLength(); i++)
		m_pBVH->setTransform(n.getIndex() + i, mat);
	n.Invalidate();
	recalculateAreaLights(n);
}

bool e_DynamicScene::UpdateScene()
{
	m_pTerrain->UpdateInvalidated();
	m_pNodeStream->UpdateInvalidated();
	m_pTriIntStream->UpdateInvalidated();
	m_pTriDataStream->UpdateInvalidated();
	m_pBVHStream->UpdateInvalidated();
	m_pBVHIndicesStream->UpdateInvalidated();
	m_pMaterialBuffer->UpdateInvalidatedCB(matUpdater(this));
	m_pTextureBuffer->UpdateInvalidated();
	m_pMeshBuffer->UpdateInvalidated();
	m_pAnimStream->UpdateInvalidated();
	m_pLightStream->UpdateInvalidated();
	m_pVolumes->UpdateInvalidated();
	if(m_uModified)
	{
		m_uModified = 0;
		m_pBVH->Build(m_pNodeStream->UsedElements(), m_pMeshBuffer->UsedElements());
		m_pBVH->UpdateInvalidated();
		return true;
	}
	return false;
}

void e_DynamicScene::AnimateMesh(e_StreamReference(e_Node) n, float t, unsigned int anim)
{
	m_uModified = 1;
	e_AnimatedMesh* m2 = AccessAnimatedMesh(n);
	unsigned int k;
	float l;
	m2->ComputeFrameIndex(t, anim, &k, &l);
	m2->k_ComputeState(anim, k, l, getKernelSceneData(), m_pBVHStream, m_pDeviceTmpFloats);
	getMesh(n).Invalidate();
}

e_KernelDynamicScene e_DynamicScene::getKernelSceneData(bool devicePointer) const
{
	e_KernelDynamicScene r;
	r.m_sAnimData = m_pAnimStream->getKernelData(devicePointer);
	r.m_sBVHIndexData = m_pBVHIndicesStream->getKernelData(devicePointer);
	r.m_sBVHIntData = m_pTriIntStream->getKernelData(devicePointer);
	r.m_sBVHNodeData = m_pBVHStream->getKernelData(devicePointer);
	r.m_sLightData = m_pLightStream->getKernelData(devicePointer);
	r.m_sMatData = m_pMaterialBuffer->getKernelData(devicePointer);
	r.m_sMeshData = m_pMeshBuffer->getKernelData(devicePointer);
	r.m_sNodeData = m_pNodeStream->getKernelData(devicePointer);
	r.m_sTexData = m_pTextureBuffer->getKernelData(devicePointer);
	r.m_sTriData = m_pTriDataStream->getKernelData(devicePointer);
	r.m_sVolume = e_KernelAggregateVolume(m_pVolumes, devicePointer);
	r.m_sSceneBVH = m_pBVH->getData(devicePointer);
	r.m_sTerrain = m_pTerrain->getKernelData(devicePointer);
	r.m_uEnvMapIndex = m_uEnvMapIndex;
	r.m_sBox = m_pBVH->m_sBox;
	r.m_Camera = *m_pCamera;

	unsigned int l = m_pLightStream->NumUsedElements();
	if(l < 20)
	{
		float* vals = (float*)alloca(sizeof(float) * l);
		for(unsigned int i = 0; i < l; i++)
		{
			vals[i] = 1.0f;
			r.m_uEmitterIndices[i] = i;
		}
		r.m_emitterPDF = Distribution1D<MAX_LIGHT_COUNT>(vals, l);
		r.m_uEmitterCount = l;
	}
	else
	{
		throw 1;
	}

	return r;
}

void e_DynamicScene::setTerrain(e_Terrain* T)
{
	delete m_pTerrain;
	m_pTerrain = T;
}

unsigned int e_DynamicScene::getCudaBufferSize()
{
	unsigned int i = m_pAnimStream->getSizeInBytes() +
					m_pTriDataStream->getSizeInBytes() + 
					m_pTriIntStream->getSizeInBytes() + 
					m_pBVHStream->getSizeInBytes() + 
					m_pBVHIndicesStream->getSizeInBytes() + 
					m_pMaterialBuffer->getSizeInBytes() +
					m_pTextureBuffer->getSizeInBytes() + 
					m_pMeshBuffer->getSizeInBytes() +
					m_pNodeStream->getSizeInBytes() +
					m_pBVH->getSizeInBytes() +
					m_pLightStream->getSizeInBytes() +
					m_pVolumes->getSizeInBytes();
	for(unsigned int i = 0; i < m_pTextureBuffer->NumUsedElements(); i++)
		i += m_pTextureBuffer[0](i)->getBufferSize();
	i += m_pTerrain->getBufferSize();
	return i;
}

AABB e_DynamicScene::getBox(e_StreamReference(e_Node) n)
{
	AABB r = AABB::Identity();
	for(unsigned int i = 0; i < n.getLength(); i++)
		r.Enlarge(n(i)->getWorldBox(getMesh(n(i)), GetNodeTransform(n)));
	return r;
}

e_StreamReference(e_KernelLight) e_DynamicScene::createLight(e_StreamReference(e_Node) Node, const char* materialName, Spectrum& L)
{
	unsigned int mi;
	ShapeSet s = CreateShape(Node, materialName, &mi);
	unsigned int* a = &m_pMaterialBuffer->operator()(Node->m_uMaterialOffset + mi)->NodeLightIndex;
	if(*a != -1)
	{
		e_StreamReference(e_KernelLight) c = m_pLightStream->operator()(Node->m_uLightIndices[*a]);
		c->SetData(e_DiffuseLight(L, s));
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
		c->SetData(e_DiffuseLight(L, s));
		return c;
	}
}

ShapeSet e_DynamicScene::CreateShape(e_StreamReference(e_Node) Node, const char* name, unsigned int* a_Mi)
{
	e_StreamReference(e_TriIntersectorData) n[MAX_SHAPE_LENGTH];
	e_StreamReference(e_TriIntersectorData2) n2[MAX_SHAPE_LENGTH];

	e_BufferReference<e_Mesh, e_KernelMesh> m = getMesh(Node);
	unsigned int c = 0, mi = -1;
		
	for(unsigned int j = 0; j < m->m_sMatInfo.getLength(); j++)
		if(strstr(m->m_sMatInfo(j)->Name, name))
		{
			mi = j;
			break;
		}
	if(mi == -1)
		throw 1;
	if(a_Mi)
		*a_Mi = mi;

	int i = 0, e = m->m_sIntInfo.getLength();
	while(i < e)
	{
		e_StreamReference(e_TriIntersectorData) sec = m->m_sIntInfo.operator()(i);
		e_StreamReference(e_TriIntersectorData2) sec2 = m->m_sIndicesInfo.operator()(i);
		unsigned int i2 = sec2->getIndex();
		e_TriangleData* d = m->m_sTriInfo(i2);
		if(d->getMatIndex(0) == mi)//do not use Node->m_uMaterialOffset, cause mi is local...
		{
			unsigned int k = 0;
			for(; k < c; k++)
				if(n2[k]->getIndex() == i2)
					break;
			if(k == c)
			{
				n2[c] = sec2;
				n[c++] = sec;
			}
		}
		i++;
	}

	ShapeSet r = ShapeSet(n, c, GetNodeTransform(Node));
	return r;
}

void e_DynamicScene::removeLight(e_StreamReference(e_Node) Node, unsigned int mi)
{
	unsigned int a = m_pMaterialBuffer->operator()(Node->m_uMaterialOffset + mi)->NodeLightIndex;
	if(a == -1)
		return;
	unsigned int* b = Node->m_uLightIndices + a;
	m_pLightStream->dealloc(m_pLightStream->operator()(*b));
	*b = -1;
}

void e_DynamicScene::removeAllLights(e_StreamReference(e_Node) Node)
{
	e_BufferReference<e_Mesh, e_KernelMesh> m = getMesh(Node);
	for(unsigned int i = 0; i < m->m_sMatInfo.getLength(); i++)
		m_pMaterialBuffer->operator()(Node->m_uMaterialOffset + i)->NodeLightIndex = -1;
	int i = 0;
	while(i < MAX_AREALIGHT_NUM && Node->m_uLightIndices[i] != -1)
	{
		m_pLightStream->dealloc(m_pLightStream->operator()(Node->m_uLightIndices[i]));
		Node->m_uLightIndices[i++] = -1;
	}
}

void e_DynamicScene::recalculateAreaLights(e_StreamReference(e_Node) Node)
{
	int i = 0; 
	unsigned int* a = Node->m_uLightIndices;
	while(a[i] != -1)
	{
		e_StreamReference(e_KernelLight) l = m_pLightStream->operator()(a[i]);
		float4x4 mat = GetNodeTransform(Node);
		l->As<e_DiffuseLight>()->Recalculate(mat);
		m_pLightStream->Invalidate(Node->m_uLightIndices[i]);
		i++;
	}
}

void e_DynamicScene::printStatus(char* dest)
{
	sprintf(dest, "Triangle intersectors : %d/%d\nBVH nodes : %d/%d\nBVH indices : %d/%d\nMaterials : %d/%d\nTextures : %d/%d\nMeshes : %d/%d\nNodes : %d/%d\nLights : %d/%d\n"
		, m_pTriIntStream->UsedElements(), m_pTriIntStream->getLength(), m_pBVHStream->UsedElements(), m_pBVHStream->getLength(), m_pBVHIndicesStream->UsedElements(), m_pBVHIndicesStream->getLength()
		, m_pMaterialBuffer->UsedElements(), m_pMaterialBuffer->getLength(), m_pTextureBuffer->UsedElements(), m_pTextureBuffer->getLength(), m_pMeshBuffer->UsedElements(), m_pMeshBuffer->getLength()
		, m_pNodeStream->UsedElements(), m_pNodeStream->getLength(), m_pLightStream->UsedElements(), m_pLightStream->getLength());
}

e_Terrain* e_DynamicScene::getTerrain()
{
	return m_pTerrain;
}

e_StreamReference(e_VolumeRegion) e_DynamicScene::AddVolume(e_VolumeRegion& r)
{
	e_StreamReference(e_VolumeRegion) r2 = m_pVolumes->malloc(1);
	*r2.operator->() = r;
	return r2;
}

e_StreamReference(e_VolumeRegion) e_DynamicScene::getVolumes()
{
	return m_pVolumes->UsedElements();
}

AABB e_DynamicScene::getAABB(e_StreamReference(e_Node) Node, const char* name, unsigned int* a_Mi)
{
	return CreateShape(Node, name, a_Mi).getBox();
}

e_BufferReference<e_Mesh, e_KernelMesh> e_DynamicScene::getMesh(e_StreamReference(e_Node) n)
{
	return m_pMeshBuffer->operator()(n->m_uMeshIndex);
}

e_StreamReference(e_KernelMaterial) e_DynamicScene::getMats(e_StreamReference(e_Node) n)
{
	e_StreamReference(e_KernelMaterial) r = m_pMaterialBuffer->operator()(n->m_uMaterialOffset, getMesh(n)->m_sMatInfo.getLength());
	r.Invalidate();
	return r;
}

e_StreamReference(e_KernelMaterial) e_DynamicScene::getMat(e_StreamReference(e_Node) n, const char* name)
{
	e_StreamReference(e_KernelMaterial) m = getMats(n);
	for(unsigned int i = 0; i < m.getLength(); i++)
	{
		const char* a = m(i)->Name;
		if(!strcmp(a, name))
		{
			m(i).Invalidate();
			return m(i);
		}
	}
	throw 1;
}

e_StreamReference(e_KernelLight) e_DynamicScene::setEnvironementMap(const Spectrum& power, const char* file)
{
	if(m_uEnvMapIndex != -1)
	{
		//TODO
	}
	e_BufferReference<e_MIPMap, e_KernelMIPMap> m = LoadTexture(file, true);
	e_InfiniteLight l = e_InfiniteLight( m_pAnimStream, m, power, getBox(getNodes()));
	e_StreamReference(e_KernelLight) r = createLight(l);
	m_uEnvMapIndex = r.getIndex();
	return r;
}

e_SceneBVH* e_DynamicScene::getSceneBVH()
{
	if(m_uModified)
	{
		m_uModified = 0;
		m_pBVH->UpdateInvalidated();
		m_pBVH->Build(m_pNodeStream->UsedElements(), m_pMeshBuffer->UsedElements());
	}
	return m_pBVH;
}