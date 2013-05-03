#include "StdAfx.h"
#include "..\Base\FrameworkInterop.h"
#include "e_DynamicScene.h"
#include <iostream>
#include <algorithm>
#include <string>
#include "e_Terrain.h"

e_SceneInitData e_SceneInitData::CreateFor_S_SanMiguel(unsigned int a_SceneNodes, unsigned int a_Lights)
{
	return CreateForSpecificMesh(1487716 + 100, 1701833 + 100, 786998 + 100, 5905498 + 100, 255, a_Lights, a_SceneNodes);
	//return CreateForSpecificMesh(10000, 10000, 10000, 15000, 255, a_Lights);
	//return CreateForSpecificMesh(7880512, 9359209, 2341126, 28077626, 255, a_Lights);
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

e_DynamicScene::e_DynamicScene(e_SceneInitData a_Data)
{
	int nodeC = 1 << 16, tCount = 1 << 16;
	m_uModified = 1;
	m_pAnimStream = new e_DataStream<char>(a_Data.m_uSizeAnimStream);
	m_pTriDataStream = new e_DataStream<e_TriangleData>(a_Data.m_uNumTriangles);
	m_pTriIntStream = new e_DataStream<e_TriIntersectorData>(a_Data.m_uNumInt);
	m_pBVHStream = new e_DataStream<e_BVHNodeData>(a_Data.m_uNumBvhNodes);
	m_pBVHIndicesStream = new e_DataStream<int>(a_Data.m_uNumBvhIndices);
	m_pMaterialBuffer = new e_DataStream<e_KernelMaterial>(a_Data.m_uNumMaterials);
	m_pMeshBuffer = new e_CachedHostDeviceBuffer<e_Mesh, e_KernelMesh>(a_Data.m_uNumNodes, sizeof(e_AnimatedMesh));
	m_pNodeStream = new e_DataStream<e_Node>(a_Data.m_uNumNodes);
	m_pTextureBuffer = new e_CachedHostDeviceBuffer<e_Texture, e_KernelTexture>(a_Data.m_uNumTextures);
	m_pLightStream = new e_HostDeviceBuffer<e_Light, e_KernelLight>(a_Data.m_uNumLights, maxLightSize());
	m_pVolumes = new e_DataStream<e_VolumeRegion>(128);
	m_pBVH = new e_SceneBVH(a_Data.m_uNumNodes);
	//cudaMalloc(&m_pDeviceTmpFloats, sizeof(e_TmpVertex) * a_MaxTriangles * 3);
	m_pTerrain = new e_Terrain(1, make_float2(0,0), make_float2(0,0));
	unsigned int a = this->getCudaBufferSize();
	//if(a > 900 * 1024 * 1024)
	//	throw 1;
}

e_DynamicScene::e_DynamicScene(InputStream& a_In)
{

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

void e_DynamicScene::Serialize(OutputStream& a_Out)
{

}

void e_DynamicScene::UpdateMaterial(e_KernelMaterial* m)
{
	e_DataStreamReference<e_KernelMaterial> mats = m_pMaterialBuffer[0](m);
	UpdateMaterial(mats);
}

void e_DynamicScene::UpdateMaterial(e_DataStreamReference<e_KernelMaterial>& mats)
{
	struct functor
	{
		e_DynamicScene* S;
		functor(e_DynamicScene* A)
		{
			S = A;
		}
		e_HostDeviceBufferReference<e_Texture, e_KernelTexture> operator()(char* file)
		{
			return S->LoadTexture(file);
		};
	};
	for(int i = 0; i < mats.getLength(); i++)
		mats(i)->LoadTextures(functor(this));
	m_pMaterialBuffer->Invalidate(DataStreamRefresh_Buffered, mats);
}

/// <summary>Creates all directories down to the specified path</summary>
/// <param name="directory">Directory that will be created recursively</param>
/// <remarks>
///   The provided directory must not be terminated with a path separator.
/// </remarks>
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
FW::String compileFile(FW::String& f)
{
	FW::String a = FW::String("Compiled/").append(f.substring(sizeof("Scenes"))), b;
	createDirectoryRecursively(a.getDirName().getPtr());
	if(a.endsWith(".obj"))
	{
		b = a.substring(0, a.lastIndexOf('.')).append(".xmsh");
		if(fileExists(b.getPtr()))
			return b;
		OutputStream a_Out(b.getPtr());
		e_Mesh::CompileObjToBinary(f.getPtr(), a_Out);
		a_Out.Close();
	}
	else if(a.endsWith(".nif"))
	{
		b = a.substring(0, a.lastIndexOf('.')).append(".xmsh");
		if(fileExists(b.getPtr()))
			return b;
		OutputStream a_Out(b.getPtr());
		e_Mesh::CompileNifToBinary(f.getPtr(), a_Out);
		a_Out.Close();
	}
	else if(a.endsWith(".md5mesh"))
	{
		
	}
	return b;
}

e_Node* e_DynamicScene::CreateNode(const char* a_MeshFile2)
{
	m_uModified = 1;
	std::string strA(a_MeshFile2);
	std::transform(strA.begin(), strA.end(), strA.begin(), ::tolower);
	bool load;
	e_HostDeviceBufferReference<e_Mesh, e_KernelMesh> M = m_pMeshBuffer->LoadCached(strA.c_str(), &load);
	if(load)
	{
		FW::String str = compileFile(FW::String(strA.c_str()));
		InputStream I(str.getPtr());
		if(str.endsWith(".xmsh"))
			new(M(0)) e_Mesh(I, m_pTriIntStream, m_pTriDataStream, m_pBVHStream, m_pBVHIndicesStream, m_pMaterialBuffer);
		else if(str.endsWith(".xanim"))
			new(M(0)) e_AnimatedMesh(I, m_pTriIntStream, m_pTriDataStream, m_pBVHStream, m_pBVHIndicesStream, m_pMaterialBuffer, m_pAnimStream);
		I.Close();
		m_pMeshBuffer->Invalidate(M);
		e_DataStreamReference<e_KernelMaterial> mats = M(0)->getMaterialInfo();
		UpdateMaterial(mats);
	}
	else if(hasEnding(strA, std::string(".md5mesh")))
	{
		e_HostDeviceBufferReference<e_Mesh, e_KernelMesh> oldM = M;
		M = m_pMeshBuffer->malloc(1);
		((e_AnimatedMesh*)oldM(0))->CreateNewMesh((e_AnimatedMesh*)M(0), m_pTriIntStream, m_pTriDataStream, m_pBVHStream, m_pBVHIndicesStream, m_pMaterialBuffer, m_pAnimStream);
		m_pMeshBuffer->Invalidate(M);
	}
	e_DataStreamReference<e_Node> N = m_pNodeStream->malloc(1);
	new(N(0)) e_Node(M.getIndex(), M(0), strA.c_str());
	m_pNodeStream->Invalidate(DataStreamRefresh_Buffered, N);
	return N(0);
}

FW::String compileFile2(FW::String& f)
{
	FW::String a = FW::String("Compiled/").append(f.substring(sizeof("Scenes/Skyrim"))), b;
	createDirectoryRecursively(a.getDirName().getPtr());
	b = a.substring(0, a.lastIndexOf('.')).append(".xtex");
	if(fileExists(b.getPtr()))
		return b;
	OutputStream a_Out(b.getPtr());
	e_Texture::CompileToBinary(f.getPtr(), a_Out);
	a_Out.Close();
	return b;
}
e_HostDeviceBufferReference<e_Texture, e_KernelTexture> e_DynamicScene::LoadTexture(char* file)
{
	char* a_File = (char*)malloc(1024);
	ZeroMemory(a_File, 1024);
	if(!fileExists(file))
	{
		strcpy(a_File, "Scenes/Textures/");
		strcat(a_File, file);
	}
	else
	{
		memcpy(a_File, file, strlen(file));
	}
	int a = strlen(a_File);
	if(a_File[a-1] == '\n')
		a_File[a-1] = 0;
	bool load;
	e_HostDeviceBufferReference<e_Texture, e_KernelTexture> T = m_pTextureBuffer->LoadCached(a_File, &load);
	if(load)
	{
		FW::String A = compileFile2(FW::String(a_File));
		InputStream I(A.getPtr());
		new(T(0)) e_Texture(I);
		I.Close();
		T(0)->CreateKernelTexture();
		m_pTextureBuffer->Invalidate(T);
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
	m_pMaterialBuffer->UpdateInvalidated();
	m_pTextureBuffer->UpdateInvalidated();
	m_pMeshBuffer->UpdateInvalidated();
	m_pAnimStream->UpdateInvalidated();
	m_pLightStream->UpdateInvalidated();
	m_pVolumes->UpdateInvalidated();
	if(m_uModified)
	{
		m_uModified = 0;
		m_pBVH->Build(m_pNodeStream->getHost(0), m_pNodeStream->NumUsedElements());
	}
}

void e_DynamicScene::AnimateMesh(e_Node* a_Node, float t, unsigned int anim)
{
	m_uModified = 1;
	e_AnimatedMesh* a_Mesh = (e_AnimatedMesh*)a_Node->m_pMesh;
	unsigned int k;
	float l;
	a_Mesh->ComputeFrameIndex(t, anim, &k, &l);
	a_Mesh->k_ComputeState(anim, k, l, getKernelSceneData(), m_pBVHStream, m_pDeviceTmpFloats);
	m_pNodeStream->Invalidate(DataStreamRefresh_Buffered, m_pNodeStream[0](a_Node));
}

e_KernelDynamicScene e_DynamicScene::getKernelSceneData()
{
	e_KernelDynamicScene r;
	r.m_sAnimData = m_pAnimStream->getKernelData();
	r.m_sBVHIndexData = m_pBVHIndicesStream->getKernelData();
	r.m_sBVHIntData = m_pTriIntStream->getKernelData();
	r.m_sBVHNodeData = m_pBVHStream->getKernelData();
	r.m_sLightData = m_pLightStream->getKernelData();
	r.m_sMatData = m_pMaterialBuffer->getKernelData();
	r.m_sMeshData = m_pMeshBuffer->getKernelData();
	r.m_sNodeData = m_pNodeStream->getKernelData();
	r.m_sTexData = m_pTextureBuffer->getKernelData();
	r.m_sTriData = m_pTriDataStream->getKernelData();
	r.m_sVolume = e_KernelAggregateVolume(m_pVolumes->UsedElements());
	r.m_sSceneBVH = m_pBVH->getData();
	r.m_sTerrain = m_pTerrain->getKernelData();
	return r;
}

void e_DynamicScene::setTerrain(e_Terrain* T)
{
	delete m_pTerrain;
	m_pTerrain = T;
}

unsigned int e_DynamicScene::getCudaBufferSize()
{
	unsigned int i = m_pTriDataStream->getSizeInBytes() + m_pTriIntStream->getSizeInBytes() + m_pBVHStream->getSizeInBytes() + m_pBVHIndicesStream->getSizeInBytes() + m_pMaterialBuffer->getSizeInBytes() +
						m_pTextureBuffer->getSizeInBytes() + m_pMeshBuffer->getSizeInBytes() + m_pNodeStream->getSizeInBytes() + m_pAnimStream->getSizeInBytes() + m_pBVH->getSizeInBytes();
	for(unsigned int i = 0; i < m_pTextureBuffer->UsedElements(); i++)
		i += m_pTextureBuffer[0](i)->getBufferSize();
	i += m_pTerrain->getBufferSize();
	return i;
}

AABB e_DynamicScene::getAABB(e_Node* N, char* name, unsigned int* a_Mi)
{
	AABB box = AABB::Identity();
	int mi = -1;
	for(int j = 0; j < N->m_pMesh->m_sMatInfo.getLength(); j++)
		if(strstr(N->m_pMesh->m_sMatInfo(j)->Name, name))
		{
			mi = j;
			break;
		}
	if(mi == -1)
		return AABB::Identity();
	if(a_Mi)
		*a_Mi = mi;

	int i = 0, e = N->m_pMesh->m_sIntInfo.getLength() * 4;
	while(i < e)
	{
		e_TriIntersectorData* sec = (e_TriIntersectorData*)((float4*)N->m_pMesh->m_sIntInfo.operator()(0) + i);
		int* ind = N->m_pMesh->m_sIndicesInfo(i);
		if(*ind == -1)
		{
			i++;
			continue;
		}
		if(*ind < -1 || *ind >= N->m_pMesh->m_sTriInfo.getLength())
			break;
		e_TriangleData* d = N->m_pMesh->m_sTriInfo(*ind);
		if(d->getMatIndex(0) == mi)
		{
			float3 a, b, c;
			sec->getData(a, b, c);
			float4x4 mat = N->getWorldMatrix();
			box.Enlarge(mat * a);
			box.Enlarge(mat * b);
			box.Enlarge(mat * c);
		}
		i += 3;
	}
	return box;
}

e_Light* e_DynamicScene::createLight(e_Node* N, char* name, const float3& col)
{
	unsigned int mi;
	AABB box = getAABB(N, name, &mi);
	e_KernelMaterial* ref = N->m_pMesh->m_sMatInfo(mi);
	ref->Emission = col;
	this->UpdateMaterial(ref);
	return this->addDirectionalLight(box, make_float3(0,-1,0), ref->Emission);//.Transform(float4x4::Translate(0,-1,0))
}

e_Light* e_DynamicScene::createLight(e_Node* N, const float3& col, char* sourceName, char* destName)
{
	unsigned int mi;
	AABB srcBox = getAABB(N, sourceName, &mi), destBox = getAABB( N, destName);
	e_KernelMaterial* ref = N->m_pMesh->m_sMatInfo(mi);
	ref->Emission = col;
	this->UpdateMaterial(ref);
	return this->addDirectedLight(destBox, srcBox, col);
}