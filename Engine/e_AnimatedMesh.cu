#include "e_Buffer.h"
#include "e_AnimatedMesh.h"
#include "e_BVHRebuilder.h"
#include "e_TriangleData.h"
#include "e_Material.h"
#include "e_TriIntersectorData.h"

//low...high
//x=(0,1,2,3), y=(4,5,6,7)
#define SELECT(x,y,o) o < 4 ? (x >> (o * 8)) : (x >> (o * 8 - 32))
#define SCRAMBLE(x,y,a,b,c,d) unsigned int(((SELECT(x, y, d)) << 24) | ((SELECT(x, y, c) << 16)) | ((SELECT(x, y, b) << 8)) | ((SELECT(x, y, a))))

CUDA_ONLY_FUNC float4x4 d_Compute(float4x4* a_Matrices, const e_AnimatedVertex& v)
{
	float4x4 mat;
	mat.zeros();
	unsigned long long idx = v.m_cBoneIndices, wgt = v.m_fBoneWeights;
	for(int i = 0; i < 8; i++)
	{
		int j = idx & 0xff;
		float w = (wgt & 0xff) / 255.0f;
		idx >>= 8; wgt >>= 8;
		float4x4 m = a_Matrices[j];
		m = m * w;
		mat = mat + m;
	}
	return mat;
}

__global__ void g_ComputeVertices(e_TmpVertex* a_Dest, e_AnimatedVertex* a_Source, float4x4* a_Matrices, float4x4* a_Matrices2, float a_lerp, unsigned int a_VCount)
{
	unsigned int N = blockIdx.x * blockDim.x + threadIdx.x;
	if(N < a_VCount)
	{
		e_AnimatedVertex v = a_Source[N];
		float4x4 mat0 = d_Compute(a_Matrices, v);
		float4x4 mat1 = d_Compute(a_Matrices2, v);
		Vec3f v0 = mat0.TransformPoint(v.m_fVertexPos), v1 = mat1.TransformPoint(v.m_fVertexPos);
		a_Dest[N].m_fPos = math::lerp(v0, v1, a_lerp);

		Vec3f n0 = mat0.TransformDirection(v.m_fNormal), n1 = mat1.TransformDirection(v.m_fNormal);
		a_Dest[N].m_fNormal = normalize(math::lerp(n0, n1, a_lerp));

		Vec3f t0 = mat0.TransformDirection(v.m_fTangent), t1 = mat1.TransformDirection(v.m_fTangent);
		a_Dest[N].m_fTangent = normalize(math::lerp(t0, t1, a_lerp));

		Vec3f b0 = mat0.TransformDirection(v.m_fBitangent), b1 = mat1.TransformDirection(v.m_fBitangent);
		a_Dest[N].m_fBiTangent = normalize(math::lerp(b0, b1, a_lerp));
	}
}

__global__ void g_ComputeTriangles(e_TmpVertex* a_Tmp, uint3* a_TriData, e_TriangleData* a_TriData2, unsigned int a_TCount)
{
	unsigned int N = blockIdx.x * blockDim.x + threadIdx.x;
	if(N < a_TCount)
	{
		uint3 t = a_TriData[N];
#ifdef EXT_TRI
		Vec3f n = normalize(cross(a_Tmp[t.z].m_fPos - a_Tmp[t.x].m_fPos, a_Tmp[t.y].m_fPos - a_Tmp[t.x].m_fPos));
		a_TriData2[N].setData(a_Tmp[t.x].m_fPos, a_Tmp[t.y].m_fPos, a_Tmp[t.z].m_fPos,
			a_Tmp[t.x].m_fNormal, a_Tmp[t.y].m_fNormal, a_Tmp[t.z].m_fNormal);
#else
		//a_TriData2[N].m_sDeviceData.Row0.
#endif
	}
}

CUDA_DEVICE AABB g_BOX;

class AnimProvider : public ISpatialInfoProvider
{
	e_TriIntersectorData* intData;
	e_TmpVertex* vertexData;
	uint3* triData;
	unsigned int m_uNumTriangles;
public:
	AnimProvider(e_AnimatedMesh* M, e_TmpVertex* V, e_StreamReference<char> S)
		: intData(M->m_sIntInfo(0)), vertexData(V), m_uNumTriangles(M->m_sTriInfo.getLength() / 3)
	{
		triData = (uint3*)S.operator char *();
	}
	virtual AABB getBox(unsigned int idx)
	{
		uint3 t = triData[idx];
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
		uint3 t = triData[a_ObjIdx];
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
void e_AnimatedMesh::k_ComputeState(unsigned int a_Anim, unsigned int a_Frame, float a_lerp, e_Stream<e_BVHNodeData>* a_BVHNodeStream, e_TmpVertex* a_DeviceTmp, e_TmpVertex* a_HostTmp)
{
	unsigned int n = (a_Frame + 1) % m_pAnimations[a_Anim].m_pFrames.size();
	float4x4* m0 = (float4x4*)m_pAnimations[a_Anim].m_pFrames[a_Frame].m_sMatrices.getDevice();
	float4x4* m1 = (float4x4*)m_pAnimations[a_Anim].m_pFrames[n].m_sMatrices.getDevice();
	g_ComputeVertices << <k_Data.m_uVertexCount / 256 + 1, 256 >> >(a_DeviceTmp, (e_AnimatedVertex*)m_sVertices.getDevice(), m0, m1, a_lerp, k_Data.m_uVertexCount);
	ThrowCudaErrors(cudaDeviceSynchronize());
	g_ComputeTriangles << <m_sTriInfo.getLength() / 256 + 1, 256 >> >(a_DeviceTmp, (uint3*)m_sTriangles.getDevice(), m_sTriInfo.getDevice(), m_sTriInfo.getLength());
	ThrowCudaErrors(cudaDeviceSynchronize());
	ThrowCudaErrors(cudaMemcpy(a_HostTmp, a_DeviceTmp, sizeof(e_TmpVertex) * k_Data.m_uVertexCount, cudaMemcpyDeviceToHost));
	AnimProvider p(this, a_HostTmp, this->m_sTriangles);
	ThrowCudaErrors(cudaDeviceSynchronize());
	if (m_pBuilder)
		m_pBuilder->Build(&p, true);
	else m_pBuilder = new e_BVHRebuilder(this, &p);
	ThrowCudaErrors(cudaDeviceSynchronize());
	m_sTriInfo.CopyFromDevice();
	m_sNodeInfo.Invalidate();
	m_sIntInfo.Invalidate();
	m_sIndicesInfo.Invalidate();
	m_sLocalBox = m_pBuilder->getBox();
	ThrowCudaErrors(cudaDeviceSynchronize());
}