#include "StdAfx.h"
#include "AnimatedMesh.h"
#include <Base/Buffer.h>
#include <Math/Vector.h>
#include <Base/FileStream.h>
#include "TriangleData.h"
#include "Material.h"
#include "TriIntersectorData.h"
#include "SpatialStructures/BVHRebuilder.h"

namespace CudaTracerLib {

void AnimationFrame::serialize(FileOutputStream& a_Out)
{
	a_Out << (size_t)m_sHostConstructionData.size();
	a_Out.Write(&m_sHostConstructionData[0], sizeof(float4x4) * (unsigned int)m_sHostConstructionData.size());
}

void AnimationFrame::deSerialize(IInStream& a_In, Stream<char>* Buf)
{
	size_t A;
	a_In >> A;
	m_sMatrices = Buf->malloc_aligned(sizeof(float4x4) * (unsigned int)A, 16);
	a_In >> m_sMatrices;
}

void Animation::serialize(FileOutputStream& a_Out)
{
	a_Out << (size_t)m_pFrames.size();
	a_Out << m_uFrameRate;
	a_Out.Write(m_sName);
	for (unsigned int i = 0; i < m_pFrames.size(); i++)
		m_pFrames[i].serialize(a_Out);
}

void Animation::deSerialize(IInStream& a_In, Stream<char>* Buf)
{
	size_t m_uNumFrames;
	a_In >> m_uNumFrames;
	a_In >> m_uFrameRate;
	a_In.Read(m_sName);
	for (unsigned int i = 0; i < m_uNumFrames; i++)
	{
		m_pFrames.push_back(AnimationFrame());
		m_pFrames[i].deSerialize(a_In, Buf);
	}
}

AnimatedMesh::AnimatedMesh(const std::string& path, IInStream& a_In, Stream<TriIntersectorData>* a_Stream0, Stream<TriangleData>* a_Stream1, Stream<BVHNodeData>* a_Stream2, Stream<TriIntersectorData2>* a_Stream3, Stream<Material>* a_Stream4, Stream<char>* a_Stream5)
	: Mesh(path, a_In, a_Stream0, a_Stream1, a_Stream2, a_Stream3, a_Stream4, a_Stream5)
{
	m_uType = MESH_ANIMAT_TOKEN;
	a_In.Read(&k_Data, sizeof(k_Data));
	for (unsigned int i = 0; i < k_Data.m_uAnimCount; i++)
	{
		Animation A;
		A.deSerialize(a_In, a_Stream5);
		m_pAnimations.push_back(A);
	}
	m_sVertices = a_Stream5->malloc_aligned<AnimatedVertex>(sizeof(AnimatedVertex) * k_Data.m_uVertexCount);
	a_In >> m_sVertices;
	m_sTriangles = a_Stream5->malloc_aligned<uint3>(sizeof(uint3) * m_sTriInfo.getLength());
	a_In >> m_sTriangles;
	a_Stream5->UpdateInvalidated();
	m_pBuilder = 0;
}

void AnimatedMesh::FreeAnim(Stream<char>* a_Stream5)
{
	if (m_pBuilder)
		delete m_pBuilder;
	a_Stream5->dealloc(m_sVertices);
	a_Stream5->dealloc(m_sTriangles);
	m_pAnimations.~vector();
}

void AnimatedMesh::CreateNewMesh(AnimatedMesh* A, Stream<TriIntersectorData>* a_Stream0, Stream<TriangleData>* a_Stream1, Stream<BVHNodeData>* a_Stream2, Stream<TriIntersectorData2>* a_Stream3, Stream<Material>* a_Stream4, Stream<char>* a_Stream5) const
{
	A->m_uType = MESH_ANIMAT_TOKEN;
	A->m_sLocalBox = m_sLocalBox;
	A->m_sMatInfo = m_sMatInfo;
	A->m_sIndicesInfo = a_Stream3->malloc(m_sIndicesInfo, true);
	A->m_sTriInfo = a_Stream1->malloc(m_sTriInfo, true);
	A->m_sNodeInfo = a_Stream2->malloc(m_sNodeInfo, true);
	A->m_sIntInfo = a_Stream0->malloc(m_sIntInfo, true);
	A->m_pBuilder = 0;

	A->k_Data = k_Data;
	A->m_pAnimations = m_pAnimations;
	A->m_sVertices = m_sVertices;
	A->m_sTriangles = m_sTriangles;
}

class AnimProvider : public ISpatialInfoProvider
{
	TriIntersectorData* intData;
	e_KernelAnimatedMesh::e_TmpVertex* vertexData;
	Vec3u* triData;
	unsigned int m_uNumTriangles;
public:
	AnimProvider(AnimatedMesh* M, e_KernelAnimatedMesh::e_TmpVertex* V, StreamReference<char> S)
		: intData(M->m_sIntInfo(0)), vertexData(V), m_uNumTriangles(M->m_sTriInfo.getLength() / 3)
	{
		triData = (Vec3u*)S.operator char *();
	}
	virtual AABB getBox(unsigned int idx)
	{
		Vec3u t = triData[idx];
		AABB b;
		b.maxV = b.minV = vertexData[t.x].m_fPos;
		b = b.Extend(vertexData[t.y].m_fPos);
		b = b.Extend(vertexData[t.z].m_fPos);
		return b;
	}
	virtual void iterateObjects(std::function<void(unsigned int)> f)
	{
		for (unsigned int i = 0; i < m_uNumTriangles; i++)
			f(i);
	}
	virtual void setObject(unsigned int a_IntersectorIdx, unsigned int a_ObjIdx)
	{
		Vec3u t = triData[a_ObjIdx];
		intData[a_IntersectorIdx].setData(vertexData[t.x].m_fPos, vertexData[t.y].m_fPos, vertexData[t.z].m_fPos);
	}
	virtual bool SplitNode(unsigned int a_ObjIdx, int dim, float pos, AABB& lBox, AABB& rBox, const AABB& refBox) const
	{
		lBox = rBox = AABB::Identity();
		Vec3u tri = ((Vec3u*)triData)[a_ObjIdx];
		Vec3f v1 = vertexData[tri.z].m_fPos;
		for (int i = 0; i < 3; i++)
		{
			Vec3f v0 = v1;
			v1 = vertexData[tri[i]].m_fPos;
			float V0[] = { v0.x, v0.y, v0.z };
			float V1[] = { v1.x, v1.y, v1.z };
			float v0p = V0[dim];
			float v1p = V1[dim];

			// Insert vertex to the boxes it belongs to.

			if (v0p <= pos)
				lBox = lBox.Extend(v0);
			if (v0p >= pos)
				rBox = rBox.Extend(v0);

			// Edge intersects the plane => insert intersection to both boxes.

			if ((v0p < pos && v1p > pos) || (v0p > pos && v1p < pos))
			{
				Vec3f t = math::lerp(v0, v1, math::clamp01((pos - v0p) / (v1p - v0p)));
				lBox = lBox.Extend(t);
				rBox = rBox.Extend(t);
			}
		}
		lBox.maxV[dim] = pos;
		rBox.minV[dim] = pos;
		lBox = lBox.Intersect(refBox);
		rBox = rBox.Intersect(refBox);
		return true;
	}
};



void AnimatedMesh::k_ComputeState(unsigned int a_Anim, unsigned int a_Frame, float a_lerp, Stream<BVHNodeData>* a_BVHNodeStream, void* a_DeviceTmp, void* a_HostTmp)
{
	CTL_ASSERT(a_Anim < m_pAnimations.size());
	CTL_ASSERT(a_Frame < m_pAnimations[a_Anim].m_pFrames.size());
	unsigned int n = (a_Frame + 1) % m_pAnimations[a_Anim].m_pFrames.size();
	float4x4* m0 = (float4x4*)m_pAnimations[a_Anim].m_pFrames[a_Frame].m_sMatrices.getDevice();
	float4x4* m1 = (float4x4*)m_pAnimations[a_Anim].m_pFrames[n].m_sMatrices.getDevice();
	launchKernels(a_DeviceTmp, (AnimatedVertex*)m_sVertices.getDevice(), m0, m1, a_lerp, (uint3*)m_sTriangles.getDevice(), m_sTriInfo.getDevice());
	ThrowCudaErrors(cudaMemcpy(a_HostTmp, (e_KernelAnimatedMesh::e_TmpVertex*)a_DeviceTmp, sizeof(e_KernelAnimatedMesh::e_TmpVertex) * k_Data.m_uVertexCount, cudaMemcpyDeviceToHost));
	AnimProvider p(this, (e_KernelAnimatedMesh::e_TmpVertex*)a_HostTmp, this->m_sTriangles);
	ThrowCudaErrors(cudaDeviceSynchronize());
	if (m_pBuilder)
		m_pBuilder->Build(&p, true);
	else m_pBuilder = new BVHRebuilder(this, &p);
	ThrowCudaErrors(cudaDeviceSynchronize());
	m_sTriInfo.CopyFromDevice();
	m_sNodeInfo.Invalidate();
	m_sIntInfo.Invalidate();
	m_sIndicesInfo.Invalidate();
	m_sLocalBox = m_pBuilder->getBox();
	ThrowCudaErrors(cudaDeviceSynchronize());
}

}