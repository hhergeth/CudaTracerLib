#include "StdAfx.h"
#include "..\Base\FrameworkInterop.h"
#include "e_DynamicScene.h"
#include <iostream>
#include <algorithm>
#include <string>
#include "e_Terrain.h"

e_SceneInitData e_SceneInitData::CreateFor_S_SanMiguel(unsigned int a_SceneNodes, unsigned int a_Lights)
{
	e_SceneInitData r = CreateForSpecificMesh(2487716 + 100, 2701833 + 100, 2586998 + 100, 12905498 + 100, 1024, a_Lights, a_SceneNodes);
	//return CreateForSpecificMesh(10000, 10000, 10000, 15000, 255, a_Lights);
	//return CreateForSpecificMesh(7880512, 9359209, 2341126, 28077626, 255, a_Lights);
	r.m_uSizeAnimStream = 16 * 1024 * 1024;
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

e_DynamicScene::e_DynamicScene(e_SceneInitData a_Data)
{
	int nodeC = 1 << 16, tCount = 1 << 16;
	m_uModified = 1;
	m_pAnimStream = new e_Stream<char>(a_Data.m_uSizeAnimStream);
	m_pTriDataStream = new e_Stream<e_TriangleData>(a_Data.m_uNumTriangles);
	m_pTriIntStream = new e_Stream<e_TriIntersectorData>(a_Data.m_uNumInt);
	m_pBVHStream = new e_Stream<e_BVHNodeData>(a_Data.m_uNumBvhNodes);
	m_pBVHIndicesStream = new e_Stream<int>(a_Data.m_uNumBvhIndices);
	m_pMaterialBuffer = new e_Stream<e_KernelMaterial>(a_Data.m_uNumMaterials);
	m_pMeshBuffer = new e_CachedBuffer<e_Mesh, e_KernelMesh>(a_Data.m_uNumNodes, sizeof(e_AnimatedMesh));
	m_pNodeStream = new e_Stream<e_Node>(a_Data.m_uNumNodes);
	m_pTextureBuffer = new e_CachedBuffer<e_Texture, e_KernelTexture>(a_Data.m_uNumTextures);
	m_pLightStream = new e_Stream<e_KernelLight>(a_Data.m_uNumLights);
	m_pVolumes = new e_Stream<e_VolumeRegion>(128);
	m_pBVH = new e_SceneBVH(a_Data.m_uNumNodes);
	if(a_Data.m_uSizeAnimStream > 1024)
		cudaMalloc(&m_pDeviceTmpFloats, sizeof(e_TmpVertex) * (1 << 16));
	m_pTerrain = new e_Terrain(1, make_float2(0,0), make_float2(0,0));
	unsigned int a = this->getCudaBufferSize();
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

void e_DynamicScene::UpdateMaterial(e_StreamReference(e_KernelMaterial) m)
{
	struct functor
	{
		e_DynamicScene* S;
		functor(e_DynamicScene* A)
		{
			S = A;
		}
		e_BufferReference<e_Texture, e_KernelTexture> operator()(char* file)
		{
			return S->LoadTexture(file);
		};
	};
	for(int i = 0; i < m.getLength(); i++)
		m(i)->LoadTextures(functor(this));
	m.Invalidate();
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

FW::String compileFile(FW::String& f)
{
	FW::String a = FW::String("Compiled/").append(f.substring(sizeof("Scenes"))), b;
	createDirectoryRecursively(a.getDirName().getPtr());

	if(a.endsWith(".obj"))
		b = a.substring(0, a.lastIndexOf('.')).append(".xmsh");
	else if(a.endsWith(".nif"))
		b = a.substring(0, a.lastIndexOf('.')).append(".xmsh");
	else if(a.endsWith(".md5mesh"))
		b = a.substring(0, a.lastIndexOf('.')).append(".xanim");
	__int64 fs = FileSize(b.getPtr());
	if(fs && fs != -1)
		return b;

	if(a.endsWith(".obj"))
	{
		OutputStream a_Out(b.getPtr());
		e_Mesh::CompileObjToBinary(f.getPtr(), a_Out);
		a_Out.Close();
	}
	else if(a.endsWith(".nif"))
	{
		OutputStream a_Out(b.getPtr());
		e_Mesh::CompileNifToBinary(f.getPtr(), a_Out);
		a_Out.Close();
	}
	else if(a.endsWith(".md5mesh"))
	{
		OutputStream a_Out(b.getPtr());
		c_StringArray A;
		char dir[255];
		ZeroMemory(dir, sizeof(dir));
		_splitpath(f.getPtr(), 0, dir, 0, 0);
		WIN32_FIND_DATA dat;
		HANDLE hFind = FindFirstFile(FW::String(dir).append("\\*.md5anim").getPtr(), &dat);
		while(hFind != INVALID_HANDLE_VALUE)
		{
			FW::String* q = new FW::String(FW::String(dir).append(dat.cFileName));
			A((char*)q->getPtr());
			if(!FindNextFile(hFind, &dat))
				break;
		}
		e_AnimatedMesh::CompileToBinary((char*)f.getPtr(), A, a_Out);
		a_Out.Close();
	}
	return b;
}

e_StreamReference(e_Node) e_DynamicScene::CreateNode(const char* a_MeshFile2)
{
	m_uModified = 1;
	std::string strA(a_MeshFile2);
	std::transform(strA.begin(), strA.end(), strA.begin(), ::tolower);
	bool load;
	e_BufferReference<e_Mesh, e_KernelMesh> M = m_pMeshBuffer->LoadCached(strA.c_str(), &load);
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
		e_StreamReference(e_KernelMaterial) mats = M(0)->m_sMatInfo;
		UpdateMaterial(mats);
	}
	else if(hasEnding(strA, std::string(".md5mesh")))
	{
		e_BufferReference<e_Mesh, e_KernelMesh> oldM = M;
		M = m_pMeshBuffer->malloc(1);
		((e_AnimatedMesh*)oldM.operator->())->CreateNewMesh((e_AnimatedMesh*)M.operator->(), m_pTriIntStream, m_pTriDataStream, m_pBVHStream, m_pBVHIndicesStream, m_pMaterialBuffer, m_pAnimStream);
		m_pMeshBuffer->Invalidate(M);
	}
	e_StreamReference(e_Node) N = m_pNodeStream->malloc(1);
	e_StreamReference(e_KernelMaterial) m2 = m_pMaterialBuffer->malloc(M->m_sMatInfo);
	m2.Invalidate();
	new(N.operator->()) e_Node(M.getIndex(), M.operator->(), strA.c_str(), m2);
	unsigned int li[MAX_AREALIGHT_NUM];
	for(unsigned int i = 0; i < M->m_uUsedLights; i++)
	{
		ShapeSet<MAX_SHAPE_LENGTH> s = CreateShape<MAX_SHAPE_LENGTH>(N, M->m_sLights[i].MatName);
		li[i] = createLight(e_DiffuseLight(M->m_sLights[i].L, s)).getIndex();
	}
	N->setLightData(li, M->m_uUsedLights);
	N.Invalidate();
	return N;
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
e_BufferReference<e_Texture, e_KernelTexture> e_DynamicScene::LoadTexture(char* file)
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
	e_BufferReference<e_Texture, e_KernelTexture> T = m_pTextureBuffer->LoadCached(a_File, &load);
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
	r.m_sVolume = e_KernelAggregateVolume(m_pVolumes);
	r.m_sSceneBVH = m_pBVH->getData();
	r.m_sTerrain = m_pTerrain->getKernelData();
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
	unsigned int i = m_pTriDataStream->getSizeInBytes() + m_pTriIntStream->getSizeInBytes() + m_pBVHStream->getSizeInBytes() + m_pBVHIndicesStream->getSizeInBytes() + m_pMaterialBuffer->getSizeInBytes() +
						m_pTextureBuffer->getSizeInBytes() + m_pMeshBuffer->getSizeInBytes() + m_pNodeStream->getSizeInBytes() + m_pAnimStream->getSizeInBytes() + m_pBVH->getSizeInBytes();
	for(unsigned int i = 0; i < m_pTextureBuffer->NumUsedElements(); i++)
		i += m_pTextureBuffer[0](i)->getBufferSize();
	i += m_pTerrain->getBufferSize();
	return i;
}