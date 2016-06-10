#pragma once

#include "Mesh.h"
#include <vector>
#include <queue>

namespace CudaTracerLib {

//additional data besides triangle and bvh data
struct AnimatedVertex
{
	Vec3f m_fVertexPos;
	NormalizedT<Vec3f> m_fNormal;
	NormalizedT<Vec3f> m_fTangent;
	NormalizedT<Vec3f> m_fBitangent;
	unsigned long long m_cBoneIndices;
	unsigned long long m_fBoneWeights;
	AnimatedVertex()
		: m_fVertexPos(0.0f), m_fNormal(0.0f), m_fTangent(0.0f), m_fBitangent(0.0f), 
		  m_cBoneIndices(0), m_fBoneWeights(0)
	{
	}
};

//frame of an animation, either storing host data for writing to a file or storing device data for compuation
struct AnimationFrame
{
	StreamReference<char> m_sMatrices;
	std::vector<NormalizedT<OrthogonalAffineMap>> m_sHostConstructionData;

	AnimationFrame(){}

	CTL_EXPORT void serialize(FileOutputStream& a_Out);

	CTL_EXPORT void deSerialize(IInStream& a_In, Stream<char>* Buf);
};

struct Animation
{
	unsigned int m_uFrameRate;
	FixedString<128> m_sName;
	std::vector<AnimationFrame> m_pFrames;//host pointer!

	Animation()
		: m_uFrameRate(0)
	{
		
	}

	Animation(unsigned int fps, const char* name, std::vector<AnimationFrame>& frames)
		: m_uFrameRate(fps), m_sName(name), m_pFrames(frames)
	{
	}

	CTL_EXPORT void serialize(FileOutputStream& a_Out);

	CTL_EXPORT void deSerialize(IInStream& a_In, Stream<char>* Buf);
};

struct e_KernelAnimatedMesh
{
	struct e_TmpVertex
	{
		Vec3f m_fPos;
		NormalizedT<Vec3f> m_fNormal;
		NormalizedT<Vec3f> m_fTangent;
		NormalizedT<Vec3f> m_fBiTangent;
	};
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
	CTL_EXPORT AnimatedMesh(const std::string& path, IInStream& a_In, Stream<TriIntersectorData>* a_Stream0, Stream<TriangleData>* a_Stream1, Stream<BVHNodeData>* a_Stream2, Stream<TriIntersectorData2>* a_Stream3, Stream<Material>* a_Stream4, Stream<char>* a_Stream5);
	CTL_EXPORT void FreeAnim(Stream<char>* a_Stream5);
	CTL_EXPORT void k_ComputeState(unsigned int a_Anim, unsigned int a_Frame, float a_lerp, Stream<BVHNodeData>* a_BVHNodeStream, void* a_DeviceTmp, void* a_HostTmp);
	CTL_EXPORT void CreateNewMesh(AnimatedMesh* A, Stream<TriIntersectorData>* a_Stream0, Stream<TriangleData>* a_Stream1, Stream<BVHNodeData>* a_Stream2, Stream<TriIntersectorData2>* a_Stream3, Stream<Material>* a_Stream4, Stream<char>* a_Stream5) const;
	void ComputeFrameIndex(float t, unsigned int a_Anim, unsigned int* a_FrameIndex, float* a_lerp)
	{
		float a = (float)m_pAnimations[a_Anim].m_uFrameRate * t;
		if (a_lerp)
			*a_lerp = math::frac(a);
		if (a_FrameIndex)
			*a_FrameIndex = ((unsigned int)a) % m_pAnimations[a_Anim].m_pFrames.size();
	}
	unsigned int numAnimations()
	{
		return k_Data.m_uAnimCount;
	}
	std::string getAnimName(unsigned int i)
	{
		return m_pAnimations[i].m_sName;
	}
private:
	void launchKernels(void* a_DeviceTmp, AnimatedVertex* A, float4x4* m0, float4x4* m1, float a_lerp, uint3* triData, TriangleData* triData2);
};

}
