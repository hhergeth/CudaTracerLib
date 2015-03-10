#pragma once

#include "e_Mesh.h"
#include <vector>
#include <queue>

struct e_BVHLevelEntry
{
	signed int m_sParent;
	signed int m_sNode;
	signed int m_sSide;
	signed int m_sLevel;
	e_BVHLevelEntry(int p, int n, int s, int l)
	{
		m_sParent = p;
		m_sNode = n;
		m_sSide = s;
		m_sLevel = l;
	}
};

struct e_BVHHierarchy
{
	//builder data
	std::vector<e_BVHLevelEntry> m_pEntries;

	e_StreamReference(char) m_sEntries;
	unsigned int levels[32];//highest...lowest
	unsigned int m_uNumLevels;

	e_BVHHierarchy(){}

	e_BVHHierarchy(e_BVHNodeData* ref);

	unsigned int numInLevel(int level)
	{
		return levels[level + 1] - levels[level];
	}

	e_BVHLevelEntry* getLevelStartOnDevice(int level)
	{
		e_BVHLevelEntry* A = (e_BVHLevelEntry*)m_sEntries.getDevice();
		return A + levels[level];
	}

	void serialize(OutputStream& a_Out)
	{
		a_Out << (size_t)m_pEntries.size() * sizeof(e_BVHLevelEntry);
		a_Out.Write(&m_pEntries[0], (unsigned int)m_pEntries.size() * sizeof(e_BVHLevelEntry));
		a_Out.Write(levels);
		a_Out << m_uNumLevels;
	}

	void deSerialize(IInStream& a_In, e_Stream<char>* Buf)
	{
		size_t l;
		a_In >> l;
		m_sEntries = Buf->malloc((unsigned int)l);
		m_sEntries.ReadFrom(a_In);
		a_In.Read(levels, sizeof(levels));
		a_In >> m_uNumLevels;
	}
};

const int g_uMaxWeights = 8;
struct CUDA_ALIGN(16) e_AnimatedVertex
{
	CUDA_ALIGN(16) Vec3f m_fVertexPos;
	CUDA_ALIGN(16) Vec3f m_fNormal;
	CUDA_ALIGN(16) Vec3f m_fTangent;
	CUDA_ALIGN(16) Vec3f m_fBitangent;
	CUDA_ALIGN(16) unsigned char m_cBoneIndices[g_uMaxWeights];
	CUDA_ALIGN(16) unsigned char m_fBoneWeights[g_uMaxWeights];
	e_AnimatedVertex()
	{
		m_fVertexPos = Vec3f(0);
		for(int i = 0; i < g_uMaxWeights; i++)
			m_cBoneIndices[i] = m_fBoneWeights[i] = 0;
	}
};

struct CUDA_ALIGN(16) e_TmpVertex
{
	CUDA_ALIGN(16) Vec3f m_fPos;
	CUDA_ALIGN(16) Vec3f m_fNormal;
	CUDA_ALIGN(16) Vec3f m_fTangent;
	CUDA_ALIGN(16) Vec3f m_fBiTangent;
};

struct e_Frame
{
	//builder info
	float4x4* m_pMatrices;
	unsigned int m_uMatrixNum;

	e_StreamReference(char) m_sMatrices;

	e_Frame(){}

	e_Frame(float4x4* mats, int N)
	{
		m_pMatrices = mats;
		m_uMatrixNum = N;
	}

	void serialize(OutputStream& a_Out)
	{
		a_Out << (size_t)m_uMatrixNum;
		a_Out.Write(m_pMatrices, sizeof(float4x4) * m_uMatrixNum);
	}

	void deSerialize(IInStream& a_In, e_Stream<char>* Buf)
	{
		size_t A;
		a_In >> A;
		m_sMatrices = Buf->malloc(sizeof(float4x4) * (unsigned int)A);
		m_sMatrices.ReadFrom(a_In);
	}
};

struct e_Animation
{
	unsigned int m_uNumFrames;
	unsigned int m_uFrameRate;
	FixedString<32> m_sName;
	std::vector<e_Frame> m_pFrames;//host pointer!

	e_Animation(){}

	e_Animation(unsigned int fps, const char* name, std::vector<e_Frame>& frames)
		: m_uNumFrames((unsigned int)frames.size()), m_uFrameRate(fps), m_sName(name), m_pFrames(frames)
	{
	}

	void serialize(OutputStream& a_Out)
	{
		a_Out << m_uNumFrames;
		a_Out << m_uFrameRate;
		a_Out.Write(&m_sName);
		for(unsigned int i = 0; i < m_uNumFrames; i++)
			m_pFrames[i].serialize(a_Out);
	}

	void deSerialize(IInStream& a_In, e_Stream<char>* Buf)
	{
		a_In >> m_uNumFrames;
		a_In >> m_uFrameRate;
		a_In.Read(&m_sName, sizeof(m_sName));
		for(unsigned int i = 0; i < m_uNumFrames; i++)
		{
			m_pFrames.push_back(e_Frame());
			m_pFrames[i].deSerialize(a_In, Buf);
		}
	}
};

struct e_KernelAnimatedMesh
{
	unsigned int m_uVertexCount;
	unsigned int m_uJointCount;
	unsigned int m_uAnimCount;
};

struct e_KernelDynamicScene;
class e_AnimatedMesh : public e_Mesh
{
	e_KernelAnimatedMesh k_Data;
	std::vector<e_Animation> m_pAnimations;
	e_StreamReference(char) m_sVertices;
	e_StreamReference(char) m_sTriangles;
	e_BVHHierarchy m_sHierchary;
public:
	e_AnimatedMesh(IInStream& a_In, e_Stream<e_TriIntersectorData>* a_Stream0, e_Stream<e_TriangleData>* a_Stream1, e_Stream<e_BVHNodeData>* a_Stream2, e_Stream<e_TriIntersectorData2>* a_Stream3, e_Stream<e_KernelMaterial>* a_Stream4, e_Stream<char>* a_Stream5);
	static void CompileToBinary(const char* a_InputFile, std::vector<std::string>& a_Anims, OutputStream& a_Out);
	void k_ComputeState(unsigned int a_Anim, unsigned int a_Frame, float a_lerp, e_KernelDynamicScene a_Data, e_Stream<e_BVHNodeData>* a_BVHNodeStream, e_TmpVertex* a_DeviceTmp);
	void CreateNewMesh(e_AnimatedMesh* A, e_Stream<e_TriIntersectorData>* a_Stream0, e_Stream<e_TriangleData>* a_Stream1, e_Stream<e_BVHNodeData>* a_Stream2, e_Stream<e_TriIntersectorData2>* a_Stream3, e_Stream<e_KernelMaterial>* a_Stream4, e_Stream<char>* a_Stream5);
	void ComputeFrameIndex(float t, unsigned int a_Anim, unsigned int* a_FrameIndex, float* a_lerp)
	{
		float a = (float)m_pAnimations[a_Anim].m_uFrameRate * t;
		if(a_lerp)
			*a_lerp = math::frac(a);
		if(a_FrameIndex)
			*a_FrameIndex = unsigned int(a) % m_pAnimations[a_Anim].m_uNumFrames;
	}
	unsigned int numAntimations()
	{
		return k_Data.m_uAnimCount;
	}
	const char* getAnimName(unsigned int i)
	{
		return m_pAnimations[i].m_sName;
	}
};