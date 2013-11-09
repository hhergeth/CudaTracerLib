#pragma once

#include "e_Mesh.h"
#include <vector>

struct e_BVHLevelEntry
{
	signed int m_sParent;
	signed int m_sNode;
	signed int m_sSide;
	e_BVHLevelEntry(int p, int n, int s)
	{
		m_sParent = p;
		m_sNode = n;
		m_sSide = s;
	}
};

const int g_uMaxWeights = 8;
struct CUDA_ALIGN(16) e_AnimatedVertex
{
	CUDA_ALIGN(16) float3 m_fVertexPos;
	CUDA_ALIGN(16) float3 m_fNormal;
	CUDA_ALIGN(16) float3 m_fTangent;
	CUDA_ALIGN(16) float3 m_fBitangent;
	CUDA_ALIGN(16) unsigned char m_cBoneIndices[g_uMaxWeights];
	CUDA_ALIGN(16) unsigned char m_fBoneWeights[g_uMaxWeights];
	e_AnimatedVertex()
	{
		m_fVertexPos = make_float3(0);
		for(int i = 0; i < g_uMaxWeights; i++)
			m_cBoneIndices[i] = m_fBoneWeights[i] = 0;
	}
};

struct CUDA_ALIGN(16) e_TmpVertex
{
	CUDA_ALIGN(16) float3 m_fPos;
	CUDA_ALIGN(16) float3 m_fNormal;
	CUDA_ALIGN(16) float3 m_fTangent;
	CUDA_ALIGN(16) float3 m_fBiTangent;
};

struct e_Frame
{
	AABB m_sBox;
};

struct e_Animation
{
	unsigned int m_uNumFrames;
	unsigned int m_uFrameRate;
	unsigned int m_uDataOffset;
};

struct e_KernelAnimatedMesh
{
	unsigned int m_uVertexCount;
	unsigned int m_uTriangleCount;
	unsigned int m_uJointCount;
	unsigned int m_uAnimCount;
	unsigned int m_uTriDataOffset;
	unsigned int m_uBVHLevelOffset;
	unsigned int m_uBVHLevelCount;
	unsigned int m_uAnimHeaderOffset;
	unsigned int m_uAnimBodyOffset;
};

struct c_StringArray
{
	std::vector<char*> data;
	c_StringArray& operator()(char* C)
	{
		data.push_back(C);
		return *this;
	}
};

struct e_KernelDynamicScene;
class e_AnimatedMesh : public e_Mesh
{
	e_StreamReference(char) m_pOffset;
	e_KernelAnimatedMesh k_Data;

	char* BASEHOST, *BASEDEVICE;
	e_AnimatedVertex* m_pVertices;
	uint3* m_pTriangles;
	int2* m_pLevels;
	e_BVHLevelEntry* m_pLevelEntries;
	e_Animation* m_pAnimations;
	float4x4* m_pAnimData;
	template<typename T> T* TRANS(T* p)
	{
		return (T*)(((unsigned long long)p - (unsigned long long)BASEHOST) + BASEDEVICE);
	}
public:
	e_AnimatedMesh(InputStream& a_In, e_Stream<e_TriIntersectorData>* a_Stream0, e_Stream<e_TriangleData>* a_Stream1, e_Stream<e_BVHNodeData>* a_Stream2, e_Stream<e_TriIntersectorData2>* a_Stream3, e_Stream<e_KernelMaterial>* a_Stream4, e_Stream<char>* a_Stream5);
	static void CompileToBinary(const char* a_InputFile, c_StringArray& a_Anims, OutputStream& a_Out);
	void k_ComputeState(unsigned int a_Anim, unsigned int a_Frame, float a_Lerp, e_KernelDynamicScene a_Data, e_Stream<e_BVHNodeData>* a_BVHNodeStream, e_TmpVertex* a_DeviceTmp);
	void CreateNewMesh(e_AnimatedMesh* A, e_Stream<e_TriIntersectorData>* a_Stream0, e_Stream<e_TriangleData>* a_Stream1, e_Stream<e_BVHNodeData>* a_Stream2, e_Stream<e_TriIntersectorData2>* a_Stream3, e_Stream<e_KernelMaterial>* a_Stream4, e_Stream<char>* a_Stream5)
	{
		A->m_sLocalBox = m_sLocalBox;
		A->m_sMatInfo = m_sMatInfo;
		A->m_sIndicesInfo = (m_sIndicesInfo);
		A->m_sTriInfo = a_Stream1->malloc(m_sTriInfo);
		A->m_sNodeInfo = a_Stream2->malloc(m_sNodeInfo);
		A->m_sIntInfo = a_Stream0->malloc(m_sIntInfo);
		A->m_pOffset = m_pOffset;
		A->BASEHOST = BASEHOST;
		A->BASEDEVICE = BASEDEVICE;
		A->m_pVertices = m_pVertices;
		A->m_pTriangles = m_pTriangles;
		A->m_pLevels = m_pLevels;
		A->m_pLevelEntries = m_pLevelEntries;
		A->m_pAnimations = m_pAnimations;
		A->m_pAnimData = m_pAnimData;
		A->k_Data = k_Data;
	}
	void ComputeFrameIndex(float t, unsigned int a_Anim, unsigned int* a_FrameIndex, float* a_Lerp)
	{
		float a = (float)m_pAnimations[a_Anim].m_uFrameRate * t;
		if(a_Lerp)
			*a_Lerp = frac(a);
		if(a_FrameIndex)
			*a_FrameIndex = unsigned int(a) % m_pAnimations[a_Anim].m_uNumFrames;
	}
};