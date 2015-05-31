#include "StdAfx.h"
#include "e_DynamicScene.h"
#include "..\Base\StringUtils.h"
#include <iostream>
#include <string>
#include "e_AnimatedMesh.h"
#include "e_Node.h"
#include "e_FileTexture.h"
#include "e_SceneBVH.h"
#include "e_Light.h"
#include "e_Buffer.h"

#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

using namespace boost::filesystem;
using namespace boost::algorithm;

struct textureLoader
{
	e_DynamicScene* S;
	textureLoader(e_DynamicScene* A)
	{
		S = A;
	}
	e_Variable<e_KernelMIPMap> operator()(const std::string& file, e_Variable<e_KernelMIPMap>& lastVal)
	{
		if (lastVal.host)
		{
			e_BufferReference<e_MIPMap, e_KernelMIPMap> ref = S->m_pTextureBuffer->translate(lastVal);
			if (ref->m_pPath == file)
				return lastVal;
			S->m_pTextureBuffer->Release(ref->m_pPath);
		}
		return S->LoadTexture(file, true).AsVar();
	};
};

class e_DynamicScene::MatStream : public e_Stream<e_KernelMaterial>
{
	std::vector<int> refCounter;
	std::vector<e_StreamReference(e_KernelMaterial)> unusedRefs;

	struct matUpdater
	{
		textureLoader L;
		matUpdater(const textureLoader& l)
			: L(l)
		{
		}
		void operator()(e_StreamReference(e_KernelMaterial) m)
		{
			m->LoadTextures(L);
			m->bsdf.As()->Update();
		}
	};

public:
	MatStream(int L)
		: e_Stream<e_KernelMaterial>(L)
	{
		refCounter.resize(L);
		std::fill(refCounter.begin(), refCounter.end(), 1);
	}

	void UpdateMaterials(textureLoader& loader)
	{
		matUpdater upd(loader);
		UpdateInvalidatedCB(upd);
	}

	void IncrementRef(e_StreamReference(e_KernelMaterial) mats)
	{
		for (unsigned int i = 0; i < mats.getLength(); i++)
			refCounter[mats(i).getIndex()] += 1;
	}

	void DecrementRef(e_StreamReference(e_KernelMaterial) mats)
	{
		for (unsigned int i = 0; i < mats.getLength(); i++)
		{
			e_StreamReference(e_KernelMaterial) ref = mats(i);
			refCounter[ref.getIndex()] -= 1;
			if (refCounter[ref.getIndex()] == 0)
			{
				unusedRefs.push_back(ref);
				//prepare for next
				refCounter[ref.getIndex()] = 1;
			}
		}
	}

	std::vector<e_StreamReference(e_KernelMaterial)> getUnreferenced()
	{
		return unusedRefs;
	}

	void ClearUnreferenced()
	{
		for (size_t i = 0; i < unusedRefs.size(); i++)
			dealloc(unusedRefs[i]);
		unusedRefs.clear();
	}
};

template<typename T> e_StreamReference(e_KernelLight) createLight(T& val, e_Stream<e_KernelLight>* m_pLightStream)
{
	e_StreamReference(e_KernelLight) r = m_pLightStream->malloc(1);
	r()->SetData(val);
	return r();
}

e_DynamicScene::e_DynamicScene(e_Sensor* C, e_SceneInitData a_Data, const std::string& texPath, const std::string& cmpPath, const std::string& dataPath)
	: m_uEnvMapIndex(0xffffffff), m_pCamera(C), m_pCompilePath(cmpPath), m_pTexturePath(texPath)
{
	int nodeC = 1 << 16, tCount = 1 << 16;
	m_pAnimStream = new e_Stream<char>(a_Data.m_uSizeAnimStream + (a_Data.m_bSupportEnvironmentMap ? sizeof(Distribution2D<4096, 4096>) : 0));
	m_pTriDataStream = new e_Stream<e_TriangleData>(a_Data.m_uNumTriangles);
	m_pTriIntStream = new e_Stream<e_TriIntersectorData>(a_Data.m_uNumInt);
	m_pBVHStream = new e_Stream<e_BVHNodeData>(a_Data.m_uNumBvhNodes);
	m_pBVHIndicesStream = new e_Stream<e_TriIntersectorData2>(a_Data.m_uNumBvhIndices);
	m_pMaterialBuffer = new MatStream(a_Data.m_uNumMaterials);
	m_pMeshBuffer = new e_CachedBuffer<e_Mesh, e_KernelMesh>(a_Data.m_uNumMeshes, sizeof(e_AnimatedMesh));
	m_pNodeStream = new e_Stream<e_Node>(a_Data.m_uNumNodes);
	m_pTextureBuffer = new e_CachedBuffer<e_MIPMap, e_KernelMIPMap>(a_Data.m_uNumTextures);
	m_pLightStream = new e_Stream<e_KernelLight>(a_Data.m_uNumLights);
	m_pVolumes = new e_Stream<e_VolumeRegion>(128);
	m_pBVH = new e_SceneBVH(a_Data.m_uNumNodes);
	CUDA_MALLOC(&m_pDeviceTmpFloats, sizeof(e_TmpVertex) * (1 << 16));
}

void e_DynamicScene::Free()
{ 
#define DEALLOC(x) { delete x; }
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
}

e_DynamicScene::~e_DynamicScene()
{
	Free();
}

e_StreamReference(e_Node) e_DynamicScene::CreateNode(const std::string& a_Token, IInStream& in, bool force_recompile)
{
	std::string token(a_Token);
	boost::algorithm::to_lower(token);
	bool load;
	e_BufferReference<e_Mesh, e_KernelMesh> M = m_pMeshBuffer->LoadCached(token, load);
	if (load || force_recompile)
	{
		IInStream* xmshStream = 0;
		bool freeStream = false;
		if(token.find(".xmsh") == std::string::npos)
		{
			path cmpFilePath = path(m_pCompilePath) / path(a_Token).replace_extension(".xmsh");
			create_directories(cmpFilePath.parent_path());
			boost::uintmax_t si = exists(cmpFilePath) ? file_size(cmpFilePath) : 0;
			time_t cmpStamp = si != 0 ? last_write_time(cmpFilePath) : time(0);
			time_t rawStamp = exists(in.getFilePath()) ? last_write_time(in.getFilePath()) : 0;
			if (si <= 4 || rawStamp != cmpStamp)
			{
				std::cout << "Started compiling mesh : " << a_Token << "\n";
				OutputStream a_Out(cmpFilePath.string());
				e_MeshCompileType t;
				m_sCmpManager.Compile(in, token, a_Out, &t);
				a_Out.Close();
				boost::filesystem::last_write_time(cmpFilePath, rawStamp);
			}
			xmshStream = OpenFile(cmpFilePath.string());
			freeStream = true;
		}
		else xmshStream = &in;
		unsigned int t;
		*xmshStream >> t;
		if(t == (unsigned int)e_MeshCompileType::Static)
			new(M(0)) e_Mesh(token, *xmshStream, m_pTriIntStream, m_pTriDataStream, m_pBVHStream, m_pBVHIndicesStream, m_pMaterialBuffer);
		else if(t == (unsigned int)e_MeshCompileType::Animated) 
			new(M(0)) e_AnimatedMesh(token, *xmshStream, m_pTriIntStream, m_pTriDataStream, m_pBVHStream, m_pBVHIndicesStream, m_pMaterialBuffer, m_pAnimStream);
		else throw std::runtime_error("Mesh file parser error.");
		if (freeStream)
			delete xmshStream;
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
	m2.Invalidate();
	new(N.operator->()) e_Node(M.getIndex(), M.operator->(), m2);
	if (!(load || force_recompile))
		m_pMaterialBuffer->IncrementRef(m2);
	unsigned int li[MAX_AREALIGHT_NUM];
	Platform::SetMemory(li, sizeof(li));
	for(unsigned int i = 0; i < M->m_uUsedLights; i++)
	{
		ShapeSet s = CreateShape(N, M->m_sLights[i].MatName);
		li[i] = ::createLight(e_DiffuseLight(M->m_sLights[i].L, s, N.getIndex()), m_pLightStream).getIndex();
	}
	N->setLightData(li, M->m_uUsedLights);
	N.Invalidate();

	ReloadTextures();

	m_pBVH->addNode(N);

	return N;
}

e_StreamReference(e_Node) e_DynamicScene::CreateNode(const std::string& a_MeshFile2, bool force_recompile)
{
	IInStream& in = *OpenFile(a_MeshFile2);
	e_StreamReference(e_Node) n = CreateNode(boost::filesystem::path(std::string(a_MeshFile2)).filename().string(), in, force_recompile);
	in.Close();
	return n;
}

e_StreamReference(e_Node) e_DynamicScene::CreateNode(unsigned int a_TriangleCount, unsigned int a_MaterialCount)
{
	e_StreamReference(e_Node) N = m_pNodeStream->malloc(1);
	e_StreamReference(e_KernelMaterial) m2 = m_pMaterialBuffer->malloc(a_MaterialCount);
	for(unsigned int i = 0; i < a_MaterialCount; i++)
	{
		m2(i) = e_KernelMaterial("auto generated material");
		m2(i)->bsdf.SetData(diffuse(CreateTexture(Spectrum(0.5f))));
	}
	e_BufferReference<e_Mesh, e_KernelMesh> M = m_pMeshBuffer->malloc(1);
	M->m_sIndicesInfo = m_pBVHIndicesStream->malloc(1);
	M->m_sIntInfo = m_pTriIntStream->malloc(1);
	M->m_sMatInfo = m2;
	M->m_sNodeInfo = m_pBVHStream->malloc(1);
	M->m_sTriInfo = m_pTriDataStream->malloc(a_TriangleCount);
	M->m_uType = MESH_STATIC_TOKEN;
	M->m_uUsedLights = 0;
	M->m_uPath = "";
	N->setLightData(0, 0);
	new(N.operator->()) e_Node(M.getIndex(), M.operator->(), m2);
	N.Invalidate();
	m_pBVH->addNode(N);
	return N;
}

void e_DynamicScene::DeleteNode(e_StreamReference(e_Node) ref)
{
	m_sRemovedNodes.push_back(ref);
	m_pBVH->removeNode(ref);
}

e_BufferReference<e_MIPMap, e_KernelMIPMap> e_DynamicScene::LoadTexture(const std::string& file, bool a_MipMap)
{
	path rawFilePath = exists(file) ? path(file) : path(m_pTexturePath) / path(file);
	if (!exists(rawFilePath))
		return LoadTexture((std::string(m_pTexturePath) + "404.jpg").c_str(), a_MipMap);
	bool load;
	e_BufferReference<e_MIPMap, e_KernelMIPMap> T = m_pTextureBuffer->LoadCached(rawFilePath.string(), load);
	if(load)
	{
		path cmpFilePath = path(m_pCompilePath) / "Images/" / rawFilePath.filename() / ".xtex";
		create_directories(path(cmpFilePath).parent_path());
		time_t rawStamp = exists(rawFilePath) ? last_write_time(rawFilePath) : time(0);
		time_t cmpStamp = exists(cmpFilePath) ? last_write_time(cmpFilePath) : 0;
		if (cmpStamp == 0 || rawStamp != cmpStamp)
		{
			OutputStream a_Out(cmpFilePath.string().c_str());
			e_MIPMap::CompileToBinary(rawFilePath.string().c_str(), a_Out, a_MipMap);
			a_Out.Close();
			boost::filesystem::last_write_time(cmpFilePath, rawStamp);
		}
		InputStream I(cmpFilePath.string().c_str());
		new(T)e_MIPMap(rawFilePath.string(), I);
		I.Close();
		T.Invalidate();
	}
	if(!T->getKernelData().m_pDeviceData)
		throw 1;
	m_pTextureBuffer->UpdateInvalidated();
	return T;
}

void e_DynamicScene::SetNodeTransform(const float4x4& mat, e_StreamReference(e_Node) n)
{
	for(unsigned int i = 0; i < n.getLength(); i++)
		m_pBVH->setTransform(n(i), mat);
	n.Invalidate();
	recalculateAreaLights(n);
}

void e_DynamicScene::InvalidateNodesInBVH(e_StreamReference(e_Node) n)
{
	m_pBVH->invalidateNode(n);
}

void e_DynamicScene::InvalidateMeshesInBVH(e_BufferReference<e_Mesh, e_KernelMesh> m)
{
	for (unsigned int i = 0; i < m_pNodeStream->NumUsedElements(); i++)
		if (m_pNodeStream->operator()(i)->m_uMeshIndex == m.getIndex())
			InvalidateNodesInBVH(m_pNodeStream->operator()(i));
}

void e_DynamicScene::ReloadTextures()
{
	m_pMaterialBuffer->UpdateMaterials(textureLoader(this));
	m_pTextureBuffer->UpdateInvalidated();
}

bool e_DynamicScene::UpdateScene()
{
	//free material -> free textures, do not load textures twice!
	for (size_t n_idx = 0; n_idx < m_sRemovedNodes.size(); n_idx++)
	{
		e_StreamReference(e_Node) ref = m_sRemovedNodes[n_idx];
		
		m_pMeshBuffer->Release(getMesh(ref)->m_uPath);

		//remove all area lights
		removeAllLights(ref);

		//deal with material
		m_pMaterialBuffer->DecrementRef(getMats(ref));

		m_pNodeStream->dealloc(ref);
	}
	m_sRemovedNodes.clear();

	for (size_t i = 0; i < m_pMeshBuffer->m_UnusedEntries.size(); i++)
	{
		e_BufferReference<e_Mesh, e_KernelMesh> ref = m_pMeshBuffer->m_UnusedEntries[i];
		ref->Free(m_pTriIntStream, m_pTriDataStream, m_pBVHStream, m_pBVHIndicesStream, m_pMaterialBuffer);
		m_pMeshBuffer->dealloc(ref);
	}
	m_pMeshBuffer->m_UnusedEntries.clear();

	struct texUnloader
	{
		e_CachedBuffer<e_MIPMap, e_KernelMIPMap>* m_pTextureBuffer;
		texUnloader(e_CachedBuffer<e_MIPMap, e_KernelMIPMap>* m_pTextureBuffer)
			: m_pTextureBuffer(m_pTextureBuffer)
		{

		}
		void operator()(const std::string& path, e_Variable<e_KernelMIPMap> var)
		{
			m_pTextureBuffer->Release(path);
		}
	};
	texUnloader tUnloader(m_pTextureBuffer);
	std::vector<e_StreamReference(e_KernelMaterial)> unusedMats = m_pMaterialBuffer->getUnreferenced();
	for (size_t i = 0; i < unusedMats.size(); i++)
	{
		unusedMats[i]->UnloadTextures(tUnloader);
	}
	m_pMaterialBuffer->ClearUnreferenced();

	for (size_t i = 0; i < m_pTextureBuffer->m_UnusedEntries.size(); i++)
	{
		e_BufferReference<e_MIPMap, e_KernelMIPMap> ref = m_pTextureBuffer->m_UnusedEntries[i];
		ref->Free();
		m_pTextureBuffer->dealloc(ref);
	}
	m_pTextureBuffer->m_UnusedEntries.clear();

	m_pNodeStream->UpdateInvalidated();
	m_pTriIntStream->UpdateInvalidated();
	m_pTriDataStream->UpdateInvalidated();
	m_pBVHStream->UpdateInvalidated();
	m_pBVHIndicesStream->UpdateInvalidated();
	m_pMeshBuffer->UpdateInvalidated();
	m_pAnimStream->UpdateInvalidated();
	m_pLightStream->UpdateInvalidated();
	m_pVolumes->UpdateInvalidated();
	ReloadTextures();
	return m_pBVH->Build(m_pNodeStream->UsedElements(), m_pMeshBuffer->UsedElements());
}

void e_DynamicScene::AnimateMesh(e_StreamReference(e_Node) n, float t, unsigned int anim)
{
	e_AnimatedMesh* m2 = AccessAnimatedMesh(n);
	unsigned int k;
	float l;
	m2->ComputeFrameIndex(t, anim, &k, &l);
	m2->k_ComputeState(anim, k, l, getKernelSceneData(), m_pBVHStream, m_pDeviceTmpFloats);
	getMesh(n).Invalidate();
	m_pBVH->invalidateNode(n);
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
	r.m_uEnvMapIndex = m_uEnvMapIndex;
	r.m_sBox = m_pBVH->m_sBox;
	r.m_Camera = *m_pCamera;

	if (m_pLightStream->NumUsedElements() < 20)
	{
		float* vals = (float*)alloca(sizeof(float) * m_pLightStream->NumUsedElements());
		int counter = 0;
		for (unsigned int i = 0; i < m_pLightStream->NumUsedElements(); i++)
		{
			if (m_pLightStream->operator()(i)->As()->IsRemoved)
				continue;
			vals[counter] = 1;
			r.m_uEmitterIndices[counter++] = i;
		}
		r.m_emitterPDF = Distribution1D<MAX_LIGHT_COUNT>(vals, counter);
		r.m_uEmitterCount = counter;
	}
	else
	{
		throw std::runtime_error("Too many lights in scene!");
	}

	return r;
}

void e_DynamicScene::instanciateNodeMaterials(e_StreamReference(e_Node) n)
{
	e_BufferReference<e_Mesh, e_KernelMesh> M = getMesh(n);
	if (m_pMaterialBuffer->NumUsedElements() + M->m_sMatInfo.getLength() < m_pMaterialBuffer->getLength() - 1)
	{
		e_StreamReference(e_KernelMaterial) m2 = m_pMaterialBuffer->malloc(M->m_sMatInfo);
		m2.Invalidate();
		m_pMaterialBuffer->DecrementRef(getMats(n));
		n->m_uMaterialOffset = m2.getIndex();
		n->m_uInstanciatedMaterial = true;
		n.Invalidate();
	}
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
	return i;
}

AABB e_DynamicScene::getBox(e_StreamReference(e_Node) n)
{
	if (n.getIndex() == 0 && n.getLength() == m_pNodeStream->NumUsedElements() && !m_pBVH->needsBuild())
		return m_pBVH->m_sBox;
	AABB r = AABB::Identity();
	for(unsigned int i = 0; i < n.getLength(); i++)
		r.Enlarge(n(i)->getWorldBox(getMesh(n(i)), GetNodeTransform(n)));
	return r;
}

e_StreamReference(e_KernelLight) e_DynamicScene::createLight(e_StreamReference(e_Node) Node, const std::string& materialName, Spectrum& L)
{
	unsigned int mi;
	ShapeSet s = CreateShape(Node, materialName, &mi);
	e_StreamReference(e_KernelMaterial) matRef = getMats(Node)(mi);
	if (matRef->NodeLightIndex != -1)
	{
		e_StreamReference(e_KernelLight) c = m_pLightStream->operator()(Node->m_uLightIndices[matRef->NodeLightIndex]);
		c->SetData(e_DiffuseLight(L, s, Node.getIndex()));
		c.Invalidate();
		return c;
	}
	else
	{
		matRef->NodeLightIndex = Node->getNextFreeLightIndex();
		if (matRef->NodeLightIndex == -1)
			throw std::runtime_error("Node already has maximum number of area lights!");
		e_StreamReference(e_KernelLight) c = m_pLightStream->malloc(1);
		Node->m_uLightIndices[matRef->NodeLightIndex] = c.getIndex();
		c->SetData(e_DiffuseLight(L, s, Node.getIndex()));
		return c;
	}
}

ShapeSet e_DynamicScene::CreateShape(e_StreamReference(e_Node) Node, const std::string& name, unsigned int* a_Mi)
{
	e_BufferReference<e_Mesh, e_KernelMesh> m = getMesh(Node);
	unsigned int matIdx = -1;
		
	e_StreamReference(e_KernelMaterial) mInfo = m->m_sMatInfo;
	for (unsigned int j = 0; j < mInfo.getLength(); j++)
	{
		const std::string& mname = m->m_sMatInfo(j)->Name;
		if (mname == name)
		{
			matIdx = j;
			break;
		}
	}
	if (matIdx == -1)
		throw std::runtime_error("Could not find material name in mesh!");
	if(a_Mi)
		*a_Mi = matIdx;

	std::vector<e_StreamReference(e_TriIntersectorData)> n;
	std::vector<e_StreamReference(e_TriIntersectorData2)> n2;
	int i = 0, e = m->m_sIntInfo.getLength();
	while(i < e)
	{
		e_StreamReference(e_TriIntersectorData) sec = m->m_sIntInfo.operator()(i);
		e_StreamReference(e_TriIntersectorData2) sec2 = m->m_sIndicesInfo.operator()(i);
		unsigned int i2 = sec2->getIndex();
		e_TriangleData* d = m->m_sTriInfo(i2);
		if(d->getMatIndex(0) == matIdx)//do not use Node->m_uMaterialOffset, cause mi is local...
		{
			unsigned int k = 0;
			for (; k < n.size(); k++)
				if(n2[k]->getIndex() == i2)
					break;
			if (k == n.size())
			{
				n.push_back(sec);
				n2.push_back(sec2);
			}
		}
		i++;
	}

	ShapeSet r = ShapeSet(&n[0], (unsigned int)n.size(), GetNodeTransform(Node), m_pAnimStream);
	return r;
}

void e_DynamicScene::removeLight(e_StreamReference(e_Node) Node, unsigned int mi)
{
	unsigned int* a = &getMats(Node)(mi)->NodeLightIndex;
	if(*a == -1)
		return;
	unsigned int* b = Node->m_uLightIndices + *a;
	*a = -1;
	//m_pLightStream->dealloc(m_pLightStream->operator()(*b));
	m_pLightStream->operator()(*b)->As()->IsRemoved = true;
	*b = -1;
}

void e_DynamicScene::removeAllLights(e_StreamReference(e_Node) Node)
{
	e_StreamReference(e_KernelMaterial) matRef = getMats(Node);
	for (unsigned int j = 0; j < matRef.getLength(); j++)
		removeLight(Node, j);
}

void e_DynamicScene::recalculateAreaLights(e_StreamReference(e_Node) Node)
{
	int i = 0; 
	while (i < MAX_AREALIGHT_NUM && Node->m_uLightIndices[i] != -1)
	{
		e_StreamReference(e_KernelLight) l = m_pLightStream->operator()(Node->m_uLightIndices[i]);
		float4x4 mat = GetNodeTransform(Node);
		l->As<e_DiffuseLight>()->Recalculate(mat, m_pAnimStream);
		m_pLightStream->Invalidate(Node->m_uLightIndices[i]);
		i++;
	}
}

std::string e_DynamicScene::printStatus()
{
	return format("Triangle intersectors : %d/%d\nBVH nodes : %d/%d\nBVH indices : %d/%d\nMaterials : %d/%d\nTextures : %d/%d\nMeshes : %d/%d\nNodes : %d/%d\nLights : %d/%d\n"
		, m_pTriIntStream->UsedElements().getLength(), m_pTriIntStream->getLength(), m_pBVHStream->UsedElements().getLength(), m_pBVHStream->getLength(), m_pBVHIndicesStream->UsedElements().getLength(), m_pBVHIndicesStream->getLength()
		, m_pMaterialBuffer->UsedElements().getLength(), m_pMaterialBuffer->getLength(), m_pTextureBuffer->UsedElements().getLength(), m_pTextureBuffer->getLength(), m_pMeshBuffer->UsedElements().getLength(), m_pMeshBuffer->getLength()
		, m_pNodeStream->UsedElements().getLength(), m_pNodeStream->getLength(), m_pLightStream->UsedElements().getLength(), m_pLightStream->getLength());
}

e_StreamReference(e_VolumeRegion) e_DynamicScene::AddVolume(e_VolumeRegion& r)
{
	e_StreamReference(e_VolumeRegion) r2 = m_pVolumes->malloc(1);
	*r2.operator->() = r;
	return r2;
}

e_StreamReference(e_VolumeRegion) e_DynamicScene::AddVolume(int w, int h, int d, const float4x4& worldToVol, const e_PhaseFunction& p)
{
	e_StreamReference(e_VolumeRegion) r2 = m_pVolumes->malloc(1);
	e_VolumeRegion r;
	r.SetData(e_VolumeGrid(p, worldToVol, m_pAnimStream, Vec3u(w, h, d)));
	*r2.operator->() = r;
	return r2;
}

e_StreamReference(e_VolumeRegion) e_DynamicScene::AddVolume(int wA, int hA, int dA,
															int wS, int hS, int dS,
															int wL, int hL, int dL, const float4x4& worldToVol, const e_PhaseFunction& p)
{
	e_StreamReference(e_VolumeRegion) r2 = m_pVolumes->malloc(1);
	e_VolumeRegion r;
	r.SetData(e_VolumeGrid(p, worldToVol, m_pAnimStream, Vec3u(wA, hA, dA), Vec3u(wS, hS, dS), Vec3u(wL, hL, dL)));
	*r2.operator->() = r;
	return r2;
}

e_StreamReference(e_VolumeRegion) e_DynamicScene::getVolumes()
{
	return m_pVolumes->UsedElements();
}

AABB e_DynamicScene::getAABB(e_StreamReference(e_Node) Node, const std::string& name, unsigned int* a_Mi)
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

e_StreamReference(e_KernelMaterial) e_DynamicScene::getMat(e_StreamReference(e_Node) n, const std::string& name)
{
	e_StreamReference(e_KernelMaterial) m = getMats(n);
	for(unsigned int i = 0; i < m.getLength(); i++)
	{
		const std::string& a = m(i)->Name;
		if(a == name)
		{
			m(i).Invalidate();
			return m(i);
		}
	}
	throw std::runtime_error("Could not find material name in mesh!");
}

e_StreamReference(e_KernelLight) e_DynamicScene::setEnvironementMap(const Spectrum& power, const std::string& file)
{
	if(m_uEnvMapIndex != -1)
	{
		//TODO
	}
	e_BufferReference<e_MIPMap, e_KernelMIPMap> m = LoadTexture(file, true);
	e_InfiniteLight l = e_InfiniteLight( m_pAnimStream, m, power, getBox(getNodes()));
	e_StreamReference(e_KernelLight) r = ::createLight(l, m_pLightStream);
	m_uEnvMapIndex = r.getIndex();
	return r;
}

unsigned int e_DynamicScene::getTriangleCount()
{
	unsigned int r = 0;
	for (unsigned int i = 0; i < m_pNodeStream->NumUsedElements(); i++)
		r += getMesh(m_pNodeStream[0](i))->getTriangleCount();
	return r;
}

e_AnimatedMesh* e_DynamicScene::AccessAnimatedMesh(e_StreamReference(e_Node) n)
{
	e_BufferReference<e_Mesh, e_KernelMesh> m = getMesh(n);
	e_AnimatedMesh* m2 = (e_AnimatedMesh*)m.operator->();
	return m2;
}

unsigned int e_DynamicScene::getLightCount(e_StreamReference(e_Node) n)
{
	int i = 0;
	while (i < MAX_AREALIGHT_NUM && n->m_uLightIndices[i] != 0xffffffff)
		i++;
	if (i == MAX_AREALIGHT_NUM)
		throw std::runtime_error("Could not allocate light slot!");
	return i;
}

e_StreamReference(e_KernelLight) e_DynamicScene::getLight(e_StreamReference(e_Node) n, unsigned int i)
{
	return m_pLightStream->operator()(n->m_uLightIndices[i], 1);
}

float4x4 e_DynamicScene::GetNodeTransform(e_StreamReference(e_Node) n)
{
	if (n.getIndex() >= m_pNodeStream->getLength())
		throw std::runtime_error("Invalid idx!");
	return m_pBVH->getNodeTransform(n);
}

e_StreamReference(e_Node) e_DynamicScene::getNodes()
{
	return m_pNodeStream->UsedElements();
}

unsigned int e_DynamicScene::getNodeCount()
{
	return m_pNodeStream->NumUsedElements();
}

unsigned int e_DynamicScene::getLightCount()
{
	return m_pLightStream->NumUsedElements();
}

e_StreamReference(e_KernelLight) e_DynamicScene::getLights()
{
	return m_pLightStream->UsedElements();
}

unsigned int e_DynamicScene::getMaterialCount()
{
	return m_pMaterialBuffer->NumUsedElements();
}

e_StreamReference(e_KernelMaterial) e_DynamicScene::getMaterials()
{
	return m_pMaterialBuffer->UsedElements();
}

e_Stream<e_KernelMaterial>* e_DynamicScene::getMatBuffer()
{
	return m_pMaterialBuffer;
}

e_BVHNodeData* e_DynamicScene::getSceneBVHNode(unsigned int idx)
{
	return m_pBVH->getBVHNode(idx);
}