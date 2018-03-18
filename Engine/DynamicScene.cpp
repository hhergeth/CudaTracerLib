#include "StdAfx.h"
#include "DynamicScene.h"
#include <iostream>
#include <string>
#include "AnimatedMesh.h"
#include "TriangleData.h"
#include "Material.h"
#include "TriIntersectorData.h"
#include <SceneTypes/Node.h>
#include "MIPMap.h"
#include "SceneBVH.h"
#include <SceneTypes/Light.h>
#include <Base/Buffer.h>
#include<iomanip>
#include <filesystem.h>
#include <algorithm>
#include <sstream>
#include <Kernel/TraceHelper.h>

namespace CudaTracerLib {

std::string IFileManager::getDataPath()
{
	std::filesystem::path p(getCompiledMeshPath("x.obj"));
	std::filesystem::path dir = p.parent_path().parent_path();
	return dir.string();
}

struct textureLoader
{
	DynamicScene* S;
	textureLoader(DynamicScene* A)
	{
		S = A;
	}
	unsigned int operator()(const std::string& file, unsigned int& lastVal) const
	{
		if (lastVal != 0xffffffff)
		{
			BufferReference<MIPMap, KernelMIPMap> ref = S->m_pTextureBuffer->operator()(lastVal, 1);

			if (ref->m_pPath == file)
				return lastVal;
			S->m_pTextureBuffer->Release(ref->m_pPath);
            lastVal = 0xffffffff;
		}
		auto r = S->LoadTexture(file, true);
		return r.getIndex();
	};
};

class DynamicScene::MatStream : public Stream<Material>
{
	std::vector<int> refCounter;
	std::vector<StreamReference<Material>> unusedRefs;

    std::vector<size_t> updated_indices;

protected:
	virtual void reallocAfterResize()
	{
		Stream<Material>::reallocAfterResize();
		refCounter.resize(this->getBufferLength());
	}
public:
	MatStream(int L)
		: Stream<Material>(L)
	{
		refCounter.resize(L);
		std::fill(refCounter.begin(), refCounter.end(), 1);
        updated_indices.reserve(1024 * 128);
	}

	void UpdateMaterialsPhase1(const textureLoader& loader)
	{
        updated_indices.clear();
        UpdateInvalidated([&](StreamReference<Material> m)
        {
            m->LoadTextures(loader);
            updated_indices.push_back(m.getIndex());
        });
	}

    void UpdateMaterialsPhase2()
    {
        for(size_t i : updated_indices)
            this->operator()(i, 1)->bsdf.As()->Update();
        updated_indices.clear();
    }

	void IncrementRef(StreamReference<Material> mats)
	{
		for (unsigned int i = 0; i < mats.getLength(); i++)
			refCounter[mats(i).getIndex()] += 1;
	}

	void DecrementRef(StreamReference<Material> mats)
	{
		for (unsigned int i = 0; i < mats.getLength(); i++)
		{
			StreamReference<Material> ref = mats(i);
			refCounter[ref.getIndex()] -= 1;
			if (refCounter[ref.getIndex()] == 0)
			{
				unusedRefs.push_back(ref);
				//prepare for next
				refCounter[ref.getIndex()] = 1;
			}
		}
	}

	std::vector<StreamReference<Material>> getUnreferenced()
	{
		return unusedRefs;
	}

	void ClearUnreferenced()
	{
		for (size_t i = 0; i < unusedRefs.size(); i++)
			dealloc(unusedRefs[i]);
		unusedRefs.clear();
	}

	bool hasAlphaMappings()
	{
		for (auto r : *this)
			if (r->AlphaMap.used())
				return true;
		return false;
	}
};

class DynamicScene::LightStream : public Stream<Light>
{
	std::vector<float> m_lightWeights;
	//normalized pdfs for correct indices
	float* m_pDeviceLightWeights, *m_pHostLeightWeights;
protected:
	virtual void reallocAfterResize()
	{
		Stream<Light>::reallocAfterResize();
		size_t L = this->getBufferLength();
		m_lightWeights.resize(L, 1.0f);
		delete[] m_pHostLeightWeights;
		m_pHostLeightWeights = new float[L];
		CUDA_FREE(m_pDeviceLightWeights);
		CUDA_MALLOC(&m_pDeviceLightWeights, sizeof(float) * L);
	}
public:
	LightStream(int L)
		: Stream<Light>(L)
	{
		m_lightWeights = std::vector<float>(L, 1.0f);
		CUDA_MALLOC(&m_pDeviceLightWeights, sizeof(float) * L);
		m_pHostLeightWeights = new float[L];
	}
	~LightStream()
	{
		CUDA_FREE(m_pDeviceLightWeights);
		delete[] m_pHostLeightWeights;
	}

	float getWeight(StreamReference<Light> ref) const
	{
		return m_lightWeights[ref.getIndex()];
	}

	void setWeight(StreamReference<Light> ref, float f)
	{
		m_lightWeights[ref.getIndex()] = f;
	}

	void fillDeviceData(bool device, KernelDynamicScene& r)
	{
		float accum = 0;
		for (auto a : *this)
			accum += m_lightWeights[a.getIndex()];

		//not really necessary
		Platform::SetMemory(m_pHostLeightWeights, sizeof(float) * m_lightWeights.size());

		r.m_numLights = min((unsigned int)MAX_NUM_LIGHTS, (unsigned int)numElements());
		unsigned int i = 0;
		for (auto a : *this)
		{
			r.m_pLightIndices[i] = a.getIndex();
			float pdf = m_lightWeights[a.getIndex()] / accum;//normalized pdf
			m_pHostLeightWeights[a.getIndex()] = pdf;
			r.m_pLightCDF[i] = (i > 0 ? r.m_pLightCDF[i - 1] : 0.0f) + pdf;
			i++;
			if (i >= r.m_numLights)
				break;
		}
		CUDA_MEMCPY_TO_DEVICE(m_pDeviceLightWeights, m_pHostLeightWeights, sizeof(float) * r.m_numLights);
		r.m_pLightPDF = device ? m_pDeviceLightWeights : m_pHostLeightWeights;
	}
};

DynamicScene::DynamicScene(Sensor* C, SceneInitData a_Data, IFileManager* fManager)
	: m_uEnvMapIndex(UINT_MAX), m_pCamera(C), m_pHostTmpFloats(0), m_pFileManager(fManager)
{
	m_pAnimStream = new Stream<char>(a_Data.m_uSizeAnimStream + (a_Data.m_bSupportEnvironmentMap ? (4096 * 4094 * 8) : 0));
	m_pTriDataStream = new Stream<TriangleData>(a_Data.m_uNumTriangles);
	m_pTriIntStream = new Stream<TriIntersectorData>(a_Data.m_uNumInt);
	m_pBVHStream = new Stream<BVHNodeData>(a_Data.m_uNumBvhNodes);
	m_pBVHIndicesStream = new Stream<TriIntersectorData2>(a_Data.m_uNumBvhIndices);
	m_pMaterialBuffer = new MatStream(a_Data.m_uNumMaterials);
	m_pMeshBuffer = new CachedBuffer<Mesh, KernelMesh>(a_Data.m_uNumMeshes, sizeof(AnimatedMesh));
	m_pNodeStream = new Stream<Node>(a_Data.m_uNumNodes);
	m_pTextureBuffer = new CachedBuffer<MIPMap, KernelMIPMap>(a_Data.m_uNumTextures);
	m_pLightStream = new LightStream(a_Data.m_uNumLights);
	m_pVolumes = new Stream<VolumeRegion>(128);
	m_pBVH = new SceneBVH(a_Data.m_uNumNodes);
	const int L = 1024 * 16, S = L * sizeof(Vec3f) * 5;
	CUDA_MALLOC(&m_pDeviceTmpFloats, S);
	m_pHostTmpFloats = (e_TmpVertex*)malloc(S);
}

DynamicScene::~DynamicScene()
{
	if (m_pBVH == 0)
	{
		std::cout << "Trying to destruct scene multiple times!" << std::endl;
		return;
	}
#define DEALLOC(x) { delete x; x = 0;}
	DEALLOC(m_pTriDataStream)
	DEALLOC(m_pTriIntStream)
	DEALLOC(m_pBVHStream)
	DEALLOC(m_pBVHIndicesStream)
	DEALLOC(m_pMaterialBuffer)
	for (auto ref : *m_pTextureBuffer)
		ref->Free();
	DEALLOC(m_pTextureBuffer)
	for (auto ref : *m_pMeshBuffer)
		if (ref->m_uType == MESH_ANIMAT_TOKEN)
			((AnimatedMesh*)ref.operator->())->FreeAnim(m_pAnimStream);
	DEALLOC(m_pMeshBuffer)
	DEALLOC(m_pNodeStream)
	DEALLOC(m_pAnimStream)
	DEALLOC(m_pLightStream)
	DEALLOC(m_pVolumes)
	CUDA_FREE(m_pDeviceTmpFloats);
	free(m_pHostTmpFloats);
#undef DEALLOC
	delete m_pBVH;
	m_pBVH = 0;
}

//given either an absolute or relative path to a non xmsh
//compute a path "unique" for this mesh
//easiest way is to use the two parent folders
//also returns a token which is identifies this mesh and can be used to load it again
std::tuple<std::filesystem::path, std::string> get_compiled_path(const std::string token, IFileManager* man)
{
    auto p = std::filesystem::path(token);
    std::string parent1 = p.has_parent_path() ? p.parent_path().filename().string() + "/" : "";
    std::string parent2 = "";
    if (p.has_parent_path() && p.parent_path().has_parent_path())
        parent2 = p.parent_path().parent_path().filename().string() + "/";

    auto compiled_folder_can = std::filesystem::canonical(std::filesystem::path(man->getCompiledMeshPath("")));
    std::string T = parent2 + parent1 + p.filename().string();
    auto path = (compiled_folder_can / T).replace_extension(".xmsh");

    return std::make_tuple(path, T);
}

StreamReference<Node> DynamicScene::CreateNode(const std::string& a_Token, IInStream& in, bool force_recompile)
{
	std::string token = to_lower(a_Token);
    bool is_compiled = token.find(".xmsh") != std::string::npos;
	auto cmp_id = is_compiled ? std::make_tuple(std::filesystem::path(token), token) : get_compiled_path(token, m_pFileManager);
    auto compiled_path = std::get<0>(cmp_id);
    auto mesh_token = std::get<1>(cmp_id);

	//visual studio doesn't support weakly_canonical at the moment so instead create the file with zero size
	create_directories(compiled_path.parent_path());
	if (!std::filesystem::exists(compiled_path))
	{
		std::ofstream file;
		file.open(compiled_path, std::ios::out);
	}

	bool load;
	BufferReference<Mesh, KernelMesh> M = m_pMeshBuffer->LoadCached(mesh_token, load);
	if (load || force_recompile)
	{
		IInStream* xmshStream = 0;
		bool freeStream = false;
		if (!is_compiled)
		{
			auto si = exists(compiled_path) ? file_size(compiled_path) : 0;
			auto cmpStamp = si != 0 ? std::filesystem::last_write_time(compiled_path) : std::filesystem::file_time_type::clock::now();
			auto rawStamp = std::filesystem::exists(in.getFilePath()) ? std::filesystem::last_write_time(in.getFilePath()) : std::filesystem::file_time_type::clock::from_time_t(0);
			if (si <= 4 || rawStamp != cmpStamp)
			{
				std::cout << "Started compiling mesh : " << token << "\n";
				FileOutputStream a_Out(compiled_path.string());
				MeshCompileType t;
				m_sCmpManager.Compile(in, token, a_Out, &t);
				a_Out.Close();
				std::filesystem::last_write_time(compiled_path, rawStamp);
			}
			xmshStream = OpenFile(compiled_path.string());
			freeStream = true;
		}
		else
		{
			xmshStream = &in;
		}

		unsigned int t;
		*xmshStream >> t;
		if (t == (unsigned int)MeshCompileType::Static)
			new(M(0)) Mesh(mesh_token, *xmshStream, m_pTriIntStream, m_pTriDataStream, m_pBVHStream, m_pBVHIndicesStream, m_pMaterialBuffer, m_pAnimStream);
		else if (t == (unsigned int)MeshCompileType::Animated)
			new(M(0)) AnimatedMesh(mesh_token, *xmshStream, m_pTriIntStream, m_pTriDataStream, m_pBVHStream, m_pBVHIndicesStream, m_pMaterialBuffer, m_pAnimStream);
		else throw std::runtime_error("Mesh file parser error.");
		if (freeStream)
			delete xmshStream;
		m_pMeshBuffer->Invalidate(M);
		M->m_sMatInfo.Invalidate();
	}
	else if (M->m_uType == MESH_ANIMAT_TOKEN)
	{
		BufferReference<Mesh, KernelMesh> oldM = M;
		M = m_pMeshBuffer->malloc(1);
		M->m_sAreaLights = std::vector<MeshPartLight>();
		((AnimatedMesh*)oldM.operator->())->CreateNewMesh((AnimatedMesh*)M.operator->(), m_pTriIntStream, m_pTriDataStream, m_pBVHStream, m_pBVHIndicesStream, m_pMaterialBuffer, m_pAnimStream);
		m_pMeshBuffer->Invalidate(M);
	}
	StreamReference<Node> N = m_pNodeStream->malloc(1);
	StreamReference<Material> m2 = M->m_sMatInfo;
	m2.Invalidate();
	new(N.operator->()) Node(M.getIndex(), M.operator->(), m2);
	if (!(load || force_recompile))
		m_pMaterialBuffer->IncrementRef(m2);

	for (unsigned int i = 0; i < M->m_sAreaLights.size(); i++)
		CreateLight(N, M->m_sAreaLights[i].MatName, M->m_sAreaLights[i].L);
	N.Invalidate();
	ReloadTextures();
	m_pBVH->addNode(N);
	return N;
}

StreamReference<Node> DynamicScene::CreateNode(const std::string& a_MeshFile2, bool force_recompile)
{
	IInStream* in = OpenFile(a_MeshFile2);
	StreamReference<Node> n = CreateNode(a_MeshFile2, *in, force_recompile);
	delete in;
	return n;
}

StreamReference<Node> DynamicScene::CreateNode(unsigned int a_TriangleCount, unsigned int a_MaterialCount)
{
	StreamReference<Node> N = m_pNodeStream->malloc(1);
	StreamReference<Material> m2 = m_pMaterialBuffer->malloc(a_MaterialCount);
	for (unsigned int i = 0; i < a_MaterialCount; i++)
	{
		m2(i) = Material("auto generated material");
		m2(i)->bsdf.SetData(diffuse(CreateTexture(Spectrum(0.5f))));
	}
	BufferReference<Mesh, KernelMesh> M = m_pMeshBuffer->malloc(1);
	M->m_sIndicesInfo = m_pBVHIndicesStream->malloc(3 * a_TriangleCount);
	M->m_sIntInfo = m_pTriIntStream->malloc(3 * a_TriangleCount);
	M->m_sMatInfo = m2;
	M->m_sNodeInfo = m_pBVHStream->malloc(3 * a_TriangleCount);
	M->m_sTriInfo = m_pTriDataStream->malloc(a_TriangleCount);
	M->m_uType = MESH_STATIC_TOKEN;
	M->m_sAreaLights = std::vector<MeshPartLight>();
	M->m_uPath = "";
	new(N.operator->()) Node(M.getIndex(), M.operator->(), m2);
	N.Invalidate();
	m_pBVH->addNode(N);
	return N;
}

void DynamicScene::DeleteNode(StreamReference<Node> ref)
{
	m_sRemovedNodes.push_back(ref);
	m_pBVH->removeNode(ref);
}

BufferReference<MIPMap, KernelMIPMap> DynamicScene::LoadTexture(const std::string& file, bool a_MipMap)
{
	std::filesystem::path rawFilePath = file;
	if (!std::filesystem::exists(rawFilePath) || std::filesystem::is_directory(rawFilePath))
		rawFilePath = m_pFileManager->getTexturePath(file);

	if (!std::filesystem::exists(rawFilePath) || std::filesystem::is_directory(rawFilePath))
	{
		std::cout << "Texture : " << file << "mapped to : " << rawFilePath << " was not found\n";
		return LoadTexture("404.jpg", a_MipMap);
	}
	bool load;
	BufferReference<MIPMap, KernelMIPMap> T = m_pTextureBuffer->LoadCached(file, load);
	if (load)
	{
		std::filesystem::path cmpFilePath(m_pFileManager->getCompiledTexturePath(rawFilePath.filename().string()));
		std::filesystem::create_directories(std::filesystem::path(cmpFilePath).parent_path());
		auto rawStamp = std::filesystem::exists(rawFilePath) ? std::filesystem::last_write_time(rawFilePath) : std::filesystem::file_time_type::clock::now();
		auto cmpStamp = std::filesystem::exists(cmpFilePath) ? std::filesystem::last_write_time(cmpFilePath) : std::filesystem::file_time_type::clock::from_time_t(0);
		if (std::filesystem::file_time_type::clock::to_time_t(cmpStamp) == 0 || rawStamp != cmpStamp)
		{
			FileOutputStream a_Out(cmpFilePath.string().c_str());
			MIPMap::CompileToBinary(rawFilePath.string().c_str(), a_Out, a_MipMap);
			a_Out.Close();
			std::filesystem::last_write_time(cmpFilePath, rawStamp);
		}
		FileInputStream I(cmpFilePath.string().c_str());
		new(T)MIPMap(file, I);
		I.Close();
		T.Invalidate();
	}
	if (!T->getKernelData().m_pDeviceData)
		throw std::runtime_error(__FUNCTION__);
	m_pTextureBuffer->UpdateInvalidated();
	return T;
}

size_t DynamicScene::enumerateLights(StreamReference<Node> node, std::function<void(StreamReference<Light>)> clb)
{
	for (size_t i = 0; i < node->m_uLights.size(); i++)
	{
		StreamReference<Light> l = m_pLightStream->operator()(node->m_uLights(i));
		clb(l);
	}
	return node->m_uLights.size();
}

void DynamicScene::SetNodeTransform(const float4x4& mat, StreamReference<Node> n)
{
	for (unsigned int i = 0; i < n.getLength(); i++)
		m_pBVH->setTransform(n(i), mat);
	n.Invalidate();
	enumerateLights(n, [&](StreamReference<Light> l)
	{
		RecomputeShape(l->As<DiffuseLight>()->shapeSet, mat);
		l.Invalidate();
	});
}

void DynamicScene::InvalidateNodesInBVH(StreamReference<Node> n)
{
	m_pBVH->invalidateNode(n);
}

void DynamicScene::InvalidateMeshesInBVH(BufferReference<Mesh, KernelMesh> m)
{
	for (Stream<Node>::iterator it = m_pNodeStream->begin(); it != m_pNodeStream->end(); ++it)
		if (it->m_uMeshIndex == m.getIndex())
			InvalidateNodesInBVH(*it);
}

void DynamicScene::ReloadTextures()
{
	m_pMaterialBuffer->UpdateMaterialsPhase1(textureLoader(this));

    //this is necessary so that textures can have access to their mipmaps
    //UpdateKernel(this);
    //same thing but not the overhead of computing the random sequence and BVH
    g_SceneDataHost.m_sMatData = m_pMaterialBuffer->getKernelData(false);
    g_SceneDataHost.m_sTexData = m_pTextureBuffer->getKernelData(false);

    m_pMaterialBuffer->UpdateMaterialsPhase2();

	textureLoader t(this);
	m_pLightStream->UpdateInvalidated([&](StreamReference<Light> l)
	{
		if (l->Is<DiffuseLight>() && l->As<DiffuseLight>()->m_rad_texture.Is<ImageTexture>())
			l->As<DiffuseLight>()->m_rad_texture.As<ImageTexture>()->LoadTextures(t);
		l->As()->Update();
	});

	m_pTextureBuffer->UpdateInvalidated();
}

bool DynamicScene::UpdateScene()
{
	//free material -> free textures, do not load textures twice!
	for (size_t n_idx = 0; n_idx < m_sRemovedNodes.size(); n_idx++)
	{
		StreamReference<Node> ref = m_sRemovedNodes[n_idx];

		m_pMeshBuffer->Release(getMesh(ref)->m_uPath);

		//remove all area lights
		removeAllLights(ref);

		//deal with material
		m_pMaterialBuffer->DecrementRef(getMaterials(ref));

		m_pNodeStream->dealloc(ref);
	}
	m_sRemovedNodes.clear();

	for (size_t i = 0; i < m_pMeshBuffer->m_UnusedEntries.size(); i++)
	{
		BufferReference<Mesh, KernelMesh> ref = m_pMeshBuffer->m_UnusedEntries[i];
		ref->Free(m_pTriIntStream, m_pTriDataStream, m_pBVHStream, m_pBVHIndicesStream, m_pMaterialBuffer);
		if (ref->m_uType == MESH_ANIMAT_TOKEN)
			((AnimatedMesh*)ref.operator->())->FreeAnim(m_pAnimStream);
		m_pMeshBuffer->dealloc(ref);
	}
	m_pMeshBuffer->m_UnusedEntries.clear();

	struct texUnloader
	{
		CachedBuffer<MIPMap, KernelMIPMap>* m_pTextureBuffer;
		texUnloader(CachedBuffer<MIPMap, KernelMIPMap>* m_pTextureBuffer)
			: m_pTextureBuffer(m_pTextureBuffer)
		{

		}
		void operator()(const std::string& path, unsigned int tex_idx)
		{
            if (tex_idx == 0xffffffff)
                return;
			m_pTextureBuffer->Release(path);
		}
	};
	texUnloader tUnloader(m_pTextureBuffer);
	std::vector<StreamReference<Material>> unusedMats = m_pMaterialBuffer->getUnreferenced();
	for (size_t i = 0; i < unusedMats.size(); i++)
	{
		unusedMats[i]->UnloadTextures(tUnloader);
	}
	m_pMaterialBuffer->ClearUnreferenced();

	for (size_t i = 0; i < m_pTextureBuffer->m_UnusedEntries.size(); i++)
	{
		BufferReference<MIPMap, KernelMIPMap> ref = m_pTextureBuffer->m_UnusedEntries[i];
		ref->Free();
		m_pTextureBuffer->dealloc(ref);
	}
	m_pTextureBuffer->m_UnusedEntries.clear();

	m_psSceneBoxEnvLight = getSceneBox();
	if (m_uEnvMapIndex != UINT_MAX)
		m_pLightStream->operator()(m_uEnvMapIndex).Invalidate();

	m_pNodeStream->UpdateInvalidated();
	m_pTriIntStream->UpdateInvalidated();
	m_pTriDataStream->UpdateInvalidated();
	m_pBVHStream->UpdateInvalidated();
	m_pBVHIndicesStream->UpdateInvalidated();
	m_pMeshBuffer->UpdateInvalidated();
	m_pAnimStream->UpdateInvalidated();
	m_pVolumes->UpdateInvalidated([](StreamReference<VolumeRegion> l){l->As()->Update(); });
	ReloadTextures();
	return m_pBVH->Build(m_pNodeStream, m_pMeshBuffer);
}

void DynamicScene::AnimateMesh(StreamReference<Node> n, float t, unsigned int anim)
{
	AnimatedMesh* m2 = AccessAnimatedMesh(n);
	unsigned int k;
	float l;
	m2->ComputeFrameIndex(t, anim, &k, &l);
	m2->k_ComputeState(anim, k, l, m_pBVHStream, m_pDeviceTmpFloats, m_pHostTmpFloats);
	getMesh(n).Invalidate();
	m_pBVH->invalidateNode(n);
}

KernelDynamicScene DynamicScene::getKernelSceneData(bool devicePointer)
{
	KernelDynamicScene r;
	r.m_sAnimData = m_pAnimStream->getKernelData(devicePointer);
	r.m_sBVHIndexData = m_pBVHIndicesStream->getKernelData(devicePointer);
	r.m_sBVHIntData = m_pTriIntStream->getKernelData(devicePointer);
	r.m_sBVHNodeData = m_pBVHStream->getKernelData(devicePointer);
	r.m_sLightBuf = m_pLightStream->getKernelData(devicePointer);
	r.m_sMatData = m_pMaterialBuffer->getKernelData(devicePointer);
	r.m_sMeshData = m_pMeshBuffer->getKernelData(devicePointer);
	r.m_sNodeData = m_pNodeStream->getKernelData(devicePointer);
	r.m_sTexData = m_pTextureBuffer->getKernelData(devicePointer);
	r.m_sTriData = m_pTriDataStream->getKernelData(devicePointer);
	r.m_sVolume = KernelAggregateVolume(m_pVolumes, devicePointer);
	r.m_sSceneBVH = m_pBVH->getData(devicePointer);
	r.m_uEnvMapIndex = m_uEnvMapIndex;
	r.m_sBox = getSceneBox();
	r.m_Camera = *m_pCamera;
	m_pLightStream->fillDeviceData(devicePointer, r);
	r.doAlphaMapping = m_pMaterialBuffer->hasAlphaMappings();
    r.m_rayTraceEps = MIN_RAYTRACE_DISTANCE_RELATIVE * r.m_sBox.Size().length();
	return r;
}

void DynamicScene::instanciateNodeMaterials(StreamReference<Node> n)
{
    if (n->m_uInstanciatedMaterial)
        return;
	StreamReference<Material> newMaterials = m_pMaterialBuffer->malloc(getMesh(n)->m_sMatInfo);
    //in its current form this functionality is broken
	//m_pMaterialBuffer->DecrementRef(getMaterials(n));
	n->m_uMaterialOffset = newMaterials.getIndex();
	n->m_uInstanciatedMaterial = true;
	n.Invalidate();
    bool b;
	for (size_t i = 0; i < newMaterials.getLength(); i++)
	{
		Material* m2 = newMaterials(i);
		if (m2->AlphaMap.used() && m2->AlphaMap.tex.Is<ImageTexture>())
			m_pTextureBuffer->LoadCached(m2->AlphaMap.tex.As<ImageTexture>()->file, b);
		if (m2->HeightMap.used && m2->HeightMap.tex.Is<ImageTexture>())
			m_pTextureBuffer->LoadCached(m2->HeightMap.tex.As<ImageTexture>()->file, b);
		if (m2->NormalMap.used && m2->NormalMap.tex.Is<ImageTexture>())
			m_pTextureBuffer->LoadCached(m2->NormalMap.tex.As<ImageTexture>()->file, b);
		auto T = m2->bsdf.As()->getTextureList();
		for (auto t : T)
			if (t->Is<ImageTexture>())
				m_pTextureBuffer->LoadCached(t->As<ImageTexture>()->file, b);
	}
	ReloadTextures();
}

size_t DynamicScene::getCudaBufferSize()
{
	size_t i = m_pAnimStream->getDeviceSizeInBytes() +
		m_pTriDataStream->getDeviceSizeInBytes() +
		m_pTriIntStream->getDeviceSizeInBytes() +
		m_pBVHStream->getDeviceSizeInBytes() +
		m_pBVHIndicesStream->getDeviceSizeInBytes() +
		m_pMaterialBuffer->getDeviceSizeInBytes() +
		m_pTextureBuffer->getDeviceSizeInBytes() +
		m_pMeshBuffer->getDeviceSizeInBytes() +
		m_pNodeStream->getDeviceSizeInBytes() +
		m_pBVH->getDeviceSizeInBytes() +
		m_pLightStream->getDeviceSizeInBytes() +
		m_pVolumes->getDeviceSizeInBytes();
	for (Buffer<MIPMap, KernelMIPMap>::iterator it = m_pTextureBuffer->begin(); it != m_pTextureBuffer->end(); ++it)
		i += it->getBufferSize();
	return i;
}

std::string DynamicScene::printInfo()
{
	const int L = 40;
#define PRINT(BUF) \
		{ \
		float s = (float)BUF->getDeviceSizeInBytes(), per = s / n * 100; \
		size_t mb = (size_t)(s / (1024 * 1024));\
		str << #BUF << std::setw(L - std::string(#BUF).size()) << std::setfill(' ') << std::right << per << "%, " << mb << "[MB]\n"; \
		}
	size_t n = getCudaBufferSize();
	std::ostringstream str;
	str.precision(2);
	PRINT(m_pAnimStream);
	PRINT(m_pTriDataStream);
	PRINT(m_pTriIntStream);
	PRINT(m_pBVHStream);
	PRINT(m_pBVHIndicesStream);
	PRINT(m_pMaterialBuffer);
	PRINT(m_pTextureBuffer);
	PRINT(m_pMeshBuffer);
	PRINT(m_pNodeStream);
	PRINT(m_pBVH);
	PRINT(m_pLightStream);
	PRINT(m_pVolumes);
	size_t l = 0;
	for (Buffer<MIPMap, KernelMIPMap>::iterator it = m_pTextureBuffer->begin(); it != m_pTextureBuffer->end(); ++it)
		l += it->getBufferSize();
	float s = (float)l, per = s / (float)n * 100;
	std::string texName = "Textures";
	str << texName << std::setw(L - texName.size()) << std::setfill(' ') << std::right << per << "%, " << (s / (1024 * 1024)) << "[MB]\n";
	return str.str();
}

AABB DynamicScene::getNodeBox(StreamReference<Node> n)
{
	AABB r = AABB::Identity();
	for (unsigned int i = 0; i < n.getLength(); i++)
		r = r.Extend(n(i)->getWorldBox(getMesh(n(i)), GetNodeTransform(n(i))));
	return r;
}

AABB DynamicScene::getSceneBox()
{
	if (!m_pBVH->needsBuild())
		return m_pBVH->getSceneBox();
	AABB res = AABB::Identity();
	for (Stream<Node>::iterator it = m_pNodeStream->begin(); it != m_pNodeStream->end(); ++it)
		res = res.Extend(it->getWorldBox(getMesh(*it), GetNodeTransform(*it)));
	return res;
}

StreamReference<Light> DynamicScene::CreateLight(StreamReference<Node> Node, const std::string& materialName, Spectrum& L)
{
	unsigned int mi;
	ShapeSet s = CreateShape(Node, materialName, &mi);
	StreamReference<Material> matRef = getMaterials(Node)(mi);
	if (matRef->NodeLightIndex != -1)
	{
		StreamReference<Light> c = m_pLightStream->operator()(Node->m_uLights(matRef->NodeLightIndex));
		c->SetData(DiffuseLight(L, s, Node.getIndex()));
		c.Invalidate();
		return c;
	}
	else
	{
		if (Node->m_uLights.isFull())
			throw std::runtime_error("Node already has maximum number of area lights!");
		StreamReference<Light> c = m_pLightStream->malloc(1);
		matRef->NodeLightIndex = (unsigned int)Node->m_uLights.size();
		Node->m_uLights.push_back(c.getIndex());
		c->SetData(DiffuseLight(L, s, Node.getIndex()));
		return c;
	}
}

void DynamicScene::RecomputeShape(ShapeSet& shape, const float4x4& mat)
{
	shape.Recalculate(mat, m_pAnimStream, m_pTriIntStream, m_pTriDataStream);
}

ShapeSet DynamicScene::CreateShape(StreamReference<Node> Node, const std::string& name, unsigned int* a_Mi)
{
	BufferReference<Mesh, KernelMesh> m = getMesh(Node);
	unsigned int matIdx = -1;

	StreamReference<Material> mInfo = m->m_sMatInfo;
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
	if (a_Mi)
		*a_Mi = matIdx;

	std::vector<StreamReference<TriIntersectorData>> n;
	std::vector<StreamReference<TriIntersectorData2>> n2;
	std::vector<StreamReference<TriangleData>> n3;
	int i = 0, e = m->m_sIntInfo.getLength();
	while (i < e)
	{
		StreamReference<TriIntersectorData> sec = m->m_sIntInfo.operator()(i);
		StreamReference<TriIntersectorData2> sec2 = m->m_sIndicesInfo.operator()(i);
		unsigned int i2 = sec2->getIndex();
		StreamReference<TriangleData> d = m->m_sTriInfo(i2);
		bool clb = m_sShapeCreationClb ? m_sShapeCreationClb(m->m_sTriInfo(i2), sec) : true;
		if (d->getMatIndex(0) == matIdx && clb)//do not use Node->m_uMaterialOffset, cause mi is local...
		{
			unsigned int k = 0;
			for (; k < n.size(); k++)
				if (n2[k]->getIndex() == i2)
					break;
			if (k == n.size())
			{
				n.push_back(sec);
				n2.push_back(sec2);
				n3.push_back(d);
			}
		}
		i++;
	}

	ShapeSet r = ShapeSet(&n[0], &n3[0], (unsigned int)n.size(), GetNodeTransform(Node), m_pAnimStream, m_pTriIntStream, m_pTriDataStream);
	return r;
}

void DynamicScene::removeLight(StreamReference<Node> Node, unsigned int mi)
{
	unsigned int* a = &getMaterials(Node)(mi)->NodeLightIndex;
	if (*a == UINT_MAX)
		return;
	unsigned int& b = Node->m_uLights(*a);
	*a = UINT_MAX;
	m_pLightStream->dealloc(b, 1);
	b = UINT_MAX;
}

void DynamicScene::removeAllLights(StreamReference<Node> Node)
{
	StreamReference<Material> matRef = getMaterials(Node);
	for (unsigned int j = 0; j < matRef.getLength(); j++)
		removeLight(Node, j);
}

StreamReference<VolumeRegion> DynamicScene::CreateVolume(const VolumeRegion& r)
{
	StreamReference<VolumeRegion> r2 = m_pVolumes->malloc(1);
	*r2.operator->() = r;
	return r2;
}

StreamReference<VolumeRegion> DynamicScene::CreateVolume(int w, int h, int d, const float4x4& worldToVol, const PhaseFunction& p)
{
	StreamReference<VolumeRegion> r2 = m_pVolumes->malloc(1);
	VolumeRegion r;
	r.SetData(VolumeGrid(p, worldToVol, m_pAnimStream, Vec3u(w, h, d)));
	*r2.operator->() = r;
	return r2;
}

StreamReference<VolumeRegion> DynamicScene::CreateVolume(int wA, int hA, int dA,
	int wS, int hS, int dS,
	int wL, int hL, int dL, const float4x4& worldToVol, const PhaseFunction& p)
{
	StreamReference<VolumeRegion> r2 = m_pVolumes->malloc(1);
	VolumeRegion r;
	r.SetData(VolumeGrid(p, worldToVol, m_pAnimStream, Vec3u(wA, hA, dA), Vec3u(wS, hS, dS), Vec3u(wL, hL, dL)));
	*r2.operator->() = r;
	return r2;
}

AABB DynamicScene::getAABB(StreamReference<Node> Node, const std::string& name, unsigned int* a_Mi)
{
	return CreateShape(Node, name, a_Mi).getBox();
}

BufferReference<Mesh, KernelMesh> DynamicScene::getMesh(StreamReference<Node> n)
{
	return m_pMeshBuffer->operator()(n->m_uMeshIndex);
}

StreamReference<Material> DynamicScene::getMaterials(StreamReference<Node> n)
{
	StreamReference<Material> r = m_pMaterialBuffer->operator()(n->m_uMaterialOffset, getMesh(n)->m_sMatInfo.getLength());
	r.Invalidate();
	return r;
}

StreamReference<Material> DynamicScene::getMaterial(StreamReference<Node> n, const std::string& name)
{
	StreamReference<Material> m = getMaterials(n);
	for (unsigned int i = 0; i < m.getLength(); i++)
	{
		const std::string& a = m(i)->Name;
		if (a == name)
		{
			m(i).Invalidate();
			return m(i);
		}
	}
	throw std::runtime_error("Could not find material name in mesh!");
}

StreamReference<Light> DynamicScene::setEnvironementMap(const Spectrum& power, const std::string& file)
{
	if (m_uEnvMapIndex != -1)
	{
		//TODO
		throw std::runtime_error("Can't set environment map when it is already set!");
	}
	BufferReference<MIPMap, KernelMIPMap> m = LoadTexture(file, true);
	m_psSceneBoxEnvLight = getSceneBox();
	InfiniteLight l = InfiniteLight(m_pAnimStream, m, power, &m_psSceneBoxEnvLight);
	StreamReference<Light> r = CreateLight(l);
	m_uEnvMapIndex = r.getIndex();
	return r;
}

AnimatedMesh* DynamicScene::AccessAnimatedMesh(StreamReference<Node> n)
{
	BufferReference<Mesh, KernelMesh> m = getMesh(n);
	AnimatedMesh* m2 = (AnimatedMesh*)m.operator->();
	return m2;
}

float4x4 DynamicScene::GetNodeTransform(StreamReference<Node> n)
{
	return m_pBVH->getNodeTransform(n);
}

StreamRange<Node>& DynamicScene::getNodes()
{
	return *m_pNodeStream;
}

StreamRange<VolumeRegion>& DynamicScene::getVolumes()
{
	return *m_pVolumes;
}

StreamRange<Light>& DynamicScene::getLights()
{
	return *m_pLightStream;
}

BufferRange<MIPMap, KernelMIPMap>& DynamicScene::getTextures()
{
	return *m_pTextureBuffer;
}

BufferRange<Mesh, KernelMesh>& DynamicScene::getMeshes()
{
	return *m_pMeshBuffer;
}

StreamRange<Material>& DynamicScene::getMaterials()
{
	return *m_pMaterialBuffer;
}

unsigned int DynamicScene::getLightCount()
{
	return (unsigned int)m_pLightStream->numElements();
}

BufferReference<Light, Light> DynamicScene::CreateLight(const Light& l)
{
	StreamReference<Light> r2 = m_pLightStream->malloc(1);
	*r2.operator->() = l;
	return r2;
}

float DynamicScene::getLeightWeight(StreamReference<Light> ref) const
{
	return m_pLightStream->getWeight(ref);
}

void DynamicScene::setLeightWeight(StreamReference<Light> ref, float f) const
{
	m_pLightStream->setWeight(ref, f);
}

}
