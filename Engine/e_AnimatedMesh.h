#pragma once

#include "e_Mesh.h"
#include <vector>
#include <queue>

const int g_uMaxWeights = 8;
struct e_AnimatedVertex
{
	Vec3f m_fVertexPos;
	Vec3f m_fNormal;
	Vec3f m_fTangent;
	Vec3f m_fBitangent;
	unsigned char m_cBoneIndices[g_uMaxWeights];
	unsigned char m_fBoneWeights[g_uMaxWeights];
	e_AnimatedVertex()
	{
		m_fVertexPos = Vec3f(0);
		for(int i = 0; i < g_uMaxWeights; i++)
			m_cBoneIndices[i] = m_fBoneWeights[i] = 0;
	}
};

struct e_TmpVertex
{
	Vec3f m_fPos;
	Vec3f m_fNormal;
	Vec3f m_fTangent;
	Vec3f m_fBiTangent;
};

struct e_Frame
{
	e_StreamReference(char) m_sMatrices;
	std::vector<float4x4> m_sHostConstructionData;

	e_Frame(){}

	void serialize(OutputStream& a_Out)
	{
		a_Out << (size_t)m_sHostConstructionData.size();
		a_Out.Write(&m_sHostConstructionData[0], sizeof(float4x4) * (unsigned int)m_sHostConstructionData.size());
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
	unsigned int m_uFrameRate;
	FixedString<128> m_sName;
	std::vector<e_Frame> m_pFrames;//host pointer!

	e_Animation(){}

	e_Animation(unsigned int fps, const char* name, std::vector<e_Frame>& frames)
		: m_uFrameRate(fps), m_sName(name), m_pFrames(frames)
	{
	}

	void serialize(OutputStream& a_Out)
	{
		a_Out << (size_t)m_pFrames.size();
		a_Out << m_uFrameRate;
		a_Out.Write(m_sName);
		for (unsigned int i = 0; i < m_pFrames.size(); i++)
			m_pFrames[i].serialize(a_Out);
	}

	void deSerialize(IInStream& a_In, e_Stream<char>* Buf)
	{
		size_t m_uNumFrames;
		a_In >> m_uNumFrames;
		a_In >> m_uFrameRate;
		a_In.Read(m_sName);
		for (unsigned int i = 0; i < m_uNumFrames; i++)
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
class e_BVHRebuilder;
class e_AnimatedMesh : public e_Mesh
{
public:
	e_KernelAnimatedMesh k_Data;
	std::vector<e_Animation> m_pAnimations;
	e_StreamReference(char) m_sVertices;
	e_StreamReference(char) m_sTriangles;
	e_BVHRebuilder* m_pBuilder;
public:
	e_AnimatedMesh(const std::string& path, IInStream& a_In, e_Stream<e_TriIntersectorData>* a_Stream0, e_Stream<e_TriangleData>* a_Stream1, e_Stream<e_BVHNodeData>* a_Stream2, e_Stream<e_TriIntersectorData2>* a_Stream3, e_Stream<e_KernelMaterial>* a_Stream4, e_Stream<char>* a_Stream5);
	void k_ComputeState(unsigned int a_Anim, unsigned int a_Frame, float a_lerp, e_KernelDynamicScene a_Data, e_Stream<e_BVHNodeData>* a_BVHNodeStream, e_TmpVertex* a_DeviceTmp, e_TmpVertex* a_HostTmp);
	void CreateNewMesh(e_AnimatedMesh* A, e_Stream<e_TriIntersectorData>* a_Stream0, e_Stream<e_TriangleData>* a_Stream1, e_Stream<e_BVHNodeData>* a_Stream2, e_Stream<e_TriIntersectorData2>* a_Stream3, e_Stream<e_KernelMaterial>* a_Stream4, e_Stream<char>* a_Stream5);
	void ComputeFrameIndex(float t, unsigned int a_Anim, unsigned int* a_FrameIndex, float* a_lerp)
	{
		float a = (float)m_pAnimations[a_Anim].m_uFrameRate * t;
		if(a_lerp)
			*a_lerp = math::frac(a);
		if(a_FrameIndex)
			*a_FrameIndex = unsigned int(a) % m_pAnimations[a_Anim].m_pFrames.size();
	}
	unsigned int numAntimations()
	{
		return k_Data.m_uAnimCount;
	}
	std::string getAnimName(unsigned int i)
	{
		return m_pAnimations[i].m_sName;
	}
};