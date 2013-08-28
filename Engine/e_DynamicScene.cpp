#include "StdAfx.h"
#include "e_DynamicScene.h"
#include <iostream>
#include <algorithm>
#include <string>
#include "e_Terrain.h"
#include "..\Base\FrameworkInterop.h"

struct textureLoader
{
	e_DynamicScene* S;
	textureLoader(e_DynamicScene* A)
	{
		S = A;
	}
	e_BufferReference<e_FileTexture, e_KernelFileTexture> operator()(char* file)
	{
		return S->LoadTexture(file);
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
	e_SceneInitData r = CreateForSpecificMesh(1000000*i, 1000000*i, 1000000*i, 1000000*i, 4096 * 5, a_Lights, a_SceneNodes);
	//e_SceneInitData r = CreateForSpecificMesh(7880512, 9359209, 2341126, 28077626, 4096 * 5, a_Lights, a_SceneNodes);//san miguel
	//e_SceneInitData r = CreateForSpecificMesh(1,1,1,1,1,1,1);
	//return CreateForSpecificMesh(10000, 10000, 10000, 15000, 255, a_Lights);
	//r.m_uSizeAnimStream = 16 * 1024 * 1024;
	r.m_uSizeAnimStream = 1;
	return r;
}

bool hasEnding (std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

inline bool fileExists(const TCHAR * file)
{
   WIN32_FIND_DATA FindFileData;
   return FindFirstFile(file, &FindFileData) != INVALID_HANDLE_VALUE;
}

template<typename T> e_Stream<T>* LL(int i)
{
	e_Stream<T>* r = new e_Stream<T>(i);
	//FW::String s = FW::sprintf("%d [MB]\n", r->getSizeInBytes() / (1024 * 1024));
	//OutputDebugString(s.getPtr());
	return r;
}

e_DynamicScene::e_DynamicScene(e_Camera* C, e_SceneInitData a_Data, const char* texPath, const char* cmpPath)
	: m_sEnvMap(e_EnvironmentMap::Identity()), m_pCamera(C)
{
	m_pCompilePath = cmpPath;
	m_pTexturePath = texPath;
	int nodeC = 1 << 16, tCount = 1 << 16;
	m_uModified = 1;
	m_pAnimStream = LL<char>(a_Data.m_uSizeAnimStream);
	m_pTriDataStream = LL<e_TriangleData>(a_Data.m_uNumTriangles);
	m_pTriIntStream = LL<e_TriIntersectorData>(a_Data.m_uNumInt);
	m_pBVHStream = LL<e_BVHNodeData>(a_Data.m_uNumBvhNodes);
	m_pBVHIndicesStream = LL<int>(a_Data.m_uNumBvhIndices);
	m_pMaterialBuffer = LL<e_KernelMaterial>(a_Data.m_uNumMaterials);
	m_pMeshBuffer = new e_CachedBuffer<e_Mesh, e_KernelMesh>(a_Data.m_uNumNodes, sizeof(e_AnimatedMesh));
	m_pNodeStream = LL<e_Node>(a_Data.m_uNumNodes);
	m_pTextureBuffer = new e_CachedBuffer<e_FileTexture, e_KernelFileTexture>(a_Data.m_uNumTextures);
	m_pMIPMapBuffer = new e_CachedBuffer<e_MIPMap, e_KernelMIPMap>(a_Data.m_uNumTextures);
	m_pLightStream = LL<e_KernelLight>(a_Data.m_uNumLights);
	m_pDist2DStream = LL<Distribution2D<4096, 4096>>(1);
	m_pVolumes = LL<e_VolumeRegion>(128);
	m_pBVH = new e_SceneBVH(a_Data.m_uNumNodes);
	if(a_Data.m_uSizeAnimStream > 1024)
		cudaMalloc(&m_pDeviceTmpFloats, sizeof(e_TmpVertex) * (1 << 16));
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
	cudaFree(m_pDeviceTmpFloats);
#undef DEALLOC
	delete m_pBVH;
	delete m_pTerrain;
}

e_DynamicScene::~e_DynamicScene()
{
	Free();
}

void createDirectoryRecursively(const std::string &directory)
{
  static const std::string separators("\\/");
 
  // If the specified directory name doesn't exist, do our thing
  DWORD fileAttributes = ::GetFileAttributes(directory.c_str());
  if(fileAttributes == INVALID_FILE_ATTRIBUTES) {
 
    // Recursively do it all again for the parent directory, if any
    std::size_t slashIndex = directory.find_last_of(separators);
    if(slashIndex != std::wstring::npos) {
      createDirectoryRecursively(directory.substr(0, slashIndex));
    }
 
    // Create the last directory on the path (the recursive calls will have taken
    // care of the parent directories by now)
    BOOL result = ::CreateDirectory(directory.c_str(), nullptr);
    if(result == FALSE) {
      throw std::runtime_error("Could not create directory");
    }
 
  } else { // Specified directory name already exists as a file or directory
 
    bool isDirectoryOrJunction =
      ((fileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) ||
      ((fileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0);
 
    if(!isDirectoryOrJunction) {
      throw std::runtime_error(
        "Could not create directory because a file with the same name exists"
      );
    }
 
  }
}

__int64 FileSize(const char* name)
{
    HANDLE hFile = CreateFile(name, GENERIC_READ, 
        FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 
        FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile==INVALID_HANDLE_VALUE)
        return -1; // error condition, could call GetLastError to find out more

    LARGE_INTEGER size;
    if (!GetFileSizeEx(hFile, &size))
    {
        CloseHandle(hFile);
        return -1; // error condition, could call GetLastError to find out more
    }

    CloseHandle(hFile);
    return size.QuadPart;
}

e_StreamReference(e_Node) e_DynamicScene::CreateNode(const char* a_MeshFile2)
{
	m_uModified = 1;
	FW::String strA = FW::String(a_MeshFile2).toLower();
	bool load;
	e_BufferReference<e_Mesh, e_KernelMesh> M = m_pMeshBuffer->LoadCached(strA.getPtr(), &load);
	if(load)
	{
		FW::String t0 = FW::String(m_pCompilePath) + strA.getFileName(), cmpPath = t0.substring(0, t0.lastIndexOf('.')) + FW::String(".xmsh");
		createDirectoryRecursively(cmpPath.getDirName().getPtr());
		if(FileSize(cmpPath.getPtr()) <= 4)
		{
			OutputStream a_Out(cmpPath.getPtr());
			e_MeshCompileType t;
			m_sCmpManager.Compile(a_MeshFile2, a_Out, &t);
			a_Out.Close();
		}
		InputStream I(cmpPath.getPtr());
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
	e_StreamReference(e_KernelMaterial) m2 = m_pMaterialBuffer->malloc(M->m_sMatInfo);
	m2.Invalidate();
	new(N.operator->()) e_Node(M.getIndex(), M.operator->(), strA.getPtr(), m2);
	unsigned int li[MAX_AREALIGHT_NUM];
	ZeroMemory(li, sizeof(li));
	for(unsigned int i = 0; i < M->m_uUsedLights; i++)
	{
		ShapeSet s = CreateShape(N, M->m_sLights[i].MatName);
		li[i] = createLight(e_DiffuseLight(M->m_sLights[i].L, s)).getIndex();
	}
	N->setLightData(li, M->m_uUsedLights);
	N.Invalidate();
	return N;
}

e_BufferReference<e_FileTexture, e_KernelFileTexture> e_DynamicScene::LoadTexture(const char* file)
{
	FW::String a = fileExists(file) ? FW::String(file) : FW::String(m_pTexturePath) + FW::String(file);
	if(a.getChar(a.getLength() - 1) == '\n')
		a = a.substring(0, a.getLength() - 1);
	bool load;
	e_BufferReference<e_FileTexture, e_KernelFileTexture> T = m_pTextureBuffer->LoadCached(a.getPtr(), &load);
	if(load)
	{
		FW::String a2 = FW::String(m_pCompilePath) + "Images\\" + a.getFileName(), b = a2.substring(0, a2.lastIndexOf('.')) + ".xtex";
		createDirectoryRecursively(b.getDirName().getPtr());
		if(FileSize(b.getPtr()) <= 0)
		{
			OutputStream a_Out(b.getPtr());
			e_FileTexture::CompileToBinary(a.getPtr(), a_Out);
			a_Out.Close();
		}
		InputStream I(b.getPtr());
		new(T) e_FileTexture(I);
		I.Close();
		T->CreateKernelTexture();
		T.Invalidate();
	}
	if(!T->getKernelData().m_pDeviceData)
		throw 1;
	return T;
}

e_BufferReference<e_MIPMap, e_KernelMIPMap> e_DynamicScene::LoadMIPMap(const char* file)
{
	FW::String a = fileExists(file) ? FW::String(file) : FW::String(m_pTexturePath) + FW::String(file);
	if(a.getChar(a.getLength()) == '\n')
		a = a.substring(0, a.getLength() - 1);
	bool load;
	e_BufferReference<e_MIPMap, e_KernelMIPMap> T = m_pMIPMapBuffer->LoadCached(a.getPtr(), &load);
	if(load)
	{
		FW::String a2 = FW::String(m_pCompilePath) + "Images\\" + a.getFileName(), b = a2.substring(0, a2.lastIndexOf('.')) + ".xmip";
		createDirectoryRecursively(b.getDirName().getPtr());
		if(FileSize(b.getPtr()) <= 0)
		{
			OutputStream a_Out(b.getPtr());
			e_MIPMap::CompileToBinary(a.getPtr(), a_Out);
			a_Out.Close();
		}
		InputStream I(b.getPtr());
		new(T) e_MIPMap(I);
		I.Close();
		T->CreateKernelTexture();
		T.Invalidate();
	}
	return T;
}

void e_DynamicScene::UpdateInvalidated()
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
	m_pMIPMapBuffer->UpdateInvalidated();
	m_pDist2DStream->UpdateInvalidated();
	if(m_uModified)
	{
		m_uModified = 0;
		m_pBVH->Build(m_pNodeStream->UsedElements(), m_pMeshBuffer->UsedElements());
	}
}

void e_DynamicScene::AnimateMesh(e_StreamReference(e_Node) a_Node, float t, unsigned int anim)
{
	m_uModified = 1;
	e_BufferReference<e_Mesh, e_KernelMesh> m = m_pMeshBuffer->operator()(a_Node->m_uMeshIndex);
	e_AnimatedMesh* m2 = (e_AnimatedMesh*)m.operator->();
	unsigned int k;
	float l;
	m2->ComputeFrameIndex(t, anim, &k, &l);
	m2->k_ComputeState(anim, k, l, getKernelSceneData(), m_pBVHStream, m_pDeviceTmpFloats);
	m.Invalidate();
}

e_KernelDynamicScene e_DynamicScene::getKernelSceneData(bool devicePointer)
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
	r.m_sEnvMap = m_sEnvMap;
	r.m_sLightSelector = e_ImportantLightSelector(this, m_pCamera);
	r.m_sBox = m_pBVH->m_sBox;
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
					m_pAnimStream->getSizeInBytes() + 
					m_pBVH->getSizeInBytes() +
					m_pMIPMapBuffer->getSizeInBytes() +
					m_pLightStream->getSizeInBytes() +
					m_pDist2DStream->getSizeInBytes() +
					m_pVolumes->getSizeInBytes();
	for(unsigned int i = 0; i < m_pTextureBuffer->NumUsedElements(); i++)
		i += m_pTextureBuffer[0](i)->getBufferSize();
	i += m_pTerrain->getBufferSize();
	return i;
}

AABB e_DynamicScene::getBox(e_StreamReference(e_Node) n)
{
	return n->getWorldBox(getMesh(n));
}

e_StreamReference(e_KernelLight) e_DynamicScene::createLight(e_StreamReference(e_Node) Node, const char* materialName, float3& L)
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
	e_TriIntersectorData* n[MAX_SHAPE_LENGTH];
	unsigned int n2[MAX_SHAPE_LENGTH];
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

	ShapeSet r = ShapeSet(n, c, Node->getWorldMatrix());
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
	for(int i = 0; i < m->m_sMatInfo.getLength(); i++)
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
		float4x4 mat = Node->getWorldMatrix();
		l->As<e_DiffuseLight>()->shapeSet.Recalculate(mat);
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
	for(int i = 0; i < m.getLength(); i++)
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

e_StreamReference(e_KernelLight) e_DynamicScene::setEnvironementMap(const float3& power, const char* file)
{
	e_BufferReference<e_MIPMap, e_KernelMIPMap> m = LoadMIPMap(file);
	e_BufferReference<Distribution2D<4096, 4096>, Distribution2D<4096, 4096>> d = getDistribution2D();
	int s = sizeof(e_InfiniteLight);
	
	e_InfiniteLight l = e_InfiniteLight(power, d, m);
	e_StreamReference(e_KernelLight) r = createLight(l);
	m_sEnvMap = e_EnvironmentMap((char*)file);
	m_sEnvMap.LoadTextures(textureLoader(this));
	//return e_StreamReference(e_KernelLight)(m_pLightStream, 0, 0);
	return r;
}

bool canUse(float3& p, float3& cp, float3& cd, float f)
{
	float d = Distance(p, cp);
	float e = dot(normalize(p - cp), cd);
	e = (e + 1) / 2.0f;
	e = clamp(e, 0.5f, 1.0f);
	return d < f * e;
}
e_ImportantLightSelector::e_ImportantLightSelector(e_DynamicScene* S, e_Camera* C)
{
	ZeroMemory(m_sIndices, sizeof(m_sIndices));
	m_uCount = 0;
	unsigned int N = sizeof(m_sIndices) / sizeof(unsigned int), M = S->m_pLightStream->NumUsedElements();
	if(M < 5)
	{
		m_uCount = M;
		for(int i = 0; i < M; i++)
			m_sIndices[i] = i;
	}
	else
	{
		float sceneScale = length(C->m_sLastFrustum.Size());
		float3 dir = C->getDir(), pos = C->getPos();
		for(int i = 0; i < S->m_pLightStream->NumUsedElements(); i++)
		{
			e_StreamReference(e_KernelLight) l = S->m_pLightStream->operator()(i);
			bool use = true;
			e_DiffuseLight* L2 = (e_DiffuseLight*)l->Data;
			e_SpotLight* L3 = (e_SpotLight*)l->Data;
			switch(l->type)
			{
			case e_PointLight_TYPE:
				use = canUse(l->As<e_PointLight>()->lightPos, pos, dir, sceneScale);
				break;
			case e_DiffuseLight_TYPE:
				//use = Use(L2->shapeSet.getBox(), cd.p, proj);
				break;
			case e_SpotLight_TYPE:
				//use = Use(L3->lightPos, vp);
				break;
			}
			if(use)
				m_sIndices[m_uCount++] = i;
			if(m_uCount == N)
				break;//not great
		}
	}
}

bool e_ImportantLightSelector::Use(AABB& box, float3& p, float4x4& proj)
{
	Ray r(p, normalize(box.Center() - p));
	float a,b;
	box.Intersect(r, &a, &b);
	float4 q = proj * make_float4(0,0,a,1);
	q /= q.w;
	return abs(q.z) < 0.5f;
}

bool e_ImportantLightSelector::Use(float3& p, float4x4& vp)
{
	float4 q = vp * make_float4(p, 1);
	q = q / q.w;
	float2 pi = make_float2(q.x, q.y) / 2.0f + make_float2(0.5f);
#define ISB(f,a,b) (f >= a && f <= b)
#define ISB2(f,a,b) (ISB(f.x, a, b) && ISB(f.y, a, b))
	return (ISB2(pi,0,1));
#undef ISB
}