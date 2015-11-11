#pragma once

#include "Mesh.h"
#include <vector>
#include <queue>

namespace CudaTracerLib {

struct CUDA_ALIGN(16) AnimatedVertex
{
	Vec3f m_fVertexPos;
	Vec3f m_fNormal;
	Vec3f m_fTangent;
	Vec3f m_fBitangent;
	unsigned long long m_cBoneIndices;
	unsigned long long m_fBoneWeights;
	AnimatedVertex()
	{
		m_fVertexPos = Vec3f(0);
		m_cBoneIndices = m_fBoneWeights = 0;
	}
};

struct e_TmpVertex
{
	Vec3f m_fPos;
	Vec3f m_fNormal;
	Vec3f m_fTangent;
	Vec3f m_fBiTangent;
};

struct AnimationFrame
{
	StreamReference<char> m_sMatrices;
	std::vector<float4x4> m_sHostConstructionData;

	AnimationFrame(){}

	void serialize(FileOutputStream& a_Out);

	void deSerialize(IInStream& a_In, Stream<char>* Buf);
};

struct Animation
{
	unsigned int m_uFrameRate;
	FixedString<128> m_sName;
	std::vector<AnimationFrame> m_pFrames;//host pointer!

	Animation(){}

	Animation(unsigned int fps, const char* name, std::vector<AnimationFrame>& frames)
		: m_uFrameRate(fps), m_sName(name), m_pFrames(frames)
	{
	}

	void serialize(FileOutputStream& a_Out);

	void deSerialize(IInStream& a_In, Stream<char>* Buf);
};

struct e_KernelAnimatedMesh
{
	unsigned int m_uVertexCount;
	unsigned int m_uJointCount;
	unsigned int m_uAnimCount;
};

struct KernelDynamicScene;
class BVHRebuilder;
class AnimatedMesh : public Mesh
{
public:
	e_KernelAnimatedMesh k_Data;
	std::vector<Animation> m_pAnimations;
	StreamReference<char> m_sVertices;
	StreamReference<char> m_sTriangles;
	BVHRebuilder* m_pBuilder;
public:
	AnimatedMesh(const std::string& path, IInStream& a_In, Stream<TriIntersectorData>* a_Stream0, Stream<TriangleData>* a_Stream1, Stream<BVHNodeData>* a_Stream2, Stream<TriIntersectorData2>* a_Stream3, Stream<Material>* a_Stream4, Stream<char>* a_Stream5);
	void k_ComputeState(unsigned int a_Anim, unsigned int a_Frame, float a_lerp, Stream<BVHNodeData>* a_BVHNodeStream, e_TmpVertex* a_DeviceTmp, e_TmpVertex* a_HostTmp);
	void CreateNewMesh(AnimatedMesh* A, Stream<TriIntersectorData>* a_Stream0, Stream<TriangleData>* a_Stream1, Stream<BVHNodeData>* a_Stream2, Stream<TriIntersectorData2>* a_Stream3, Stream<Material>* a_Stream4, Stream<char>* a_Stream5);
	void ComputeFrameIndex(float t, unsigned int a_Anim, unsigned int* a_FrameIndex, float* a_lerp)
	{
		float a = (float)m_pAnimations[a_Anim].m_uFrameRate * t;
		if (a_lerp)
			*a_lerp = math::frac(a);
		if (a_FrameIndex)
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

}