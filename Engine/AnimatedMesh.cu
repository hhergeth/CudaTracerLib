#include "AnimatedMesh.h"
#include "BVHRebuilder.h"
#include "TriangleData.h"
#include "TriIntersectorData.h"

namespace CudaTracerLib {

//low...high
//x=(0,1,2,3), y=(4,5,6,7)
#define SELECT(x,y,o) o < 4 ? (x >> (o * 8)) : (x >> (o * 8 - 32))
#define SCRAMBLE(x,y,a,b,c,d) unsigned int(((SELECT(x, y, d)) << 24) | ((SELECT(x, y, c) << 16)) | ((SELECT(x, y, b) << 8)) | ((SELECT(x, y, a))))

CUDA_ONLY_FUNC float4x4 d_Compute(float4x4* a_Matrices, const AnimatedVertex& v)
{
	float4x4 mat;
	mat.zeros();
	unsigned long long idx = v.m_cBoneIndices, wgt = v.m_fBoneWeights;
	for (int i = 0; i < 8; i++)
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

__global__ void g_ComputeVertices(e_KernelAnimatedMesh::e_TmpVertex* a_Dest, AnimatedVertex* a_Source, float4x4* a_Matrices, float4x4* a_Matrices2, float a_lerp, unsigned int a_VCount)
{
	unsigned int N = blockIdx.x * blockDim.x + threadIdx.x;
	if (N < a_VCount)
	{
		AnimatedVertex v = a_Source[N];
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

__global__ void g_ComputeTriangles(e_KernelAnimatedMesh::e_TmpVertex* a_Tmp, uint3* a_TriData, TriangleData* a_TriData2, unsigned int a_TCount)
{
	unsigned int N = blockIdx.x * blockDim.x + threadIdx.x;
	if (N < a_TCount)
	{
		uint3 t = a_TriData[N];
#ifdef EXT_TRI
		Vec3f n = normalize(cross(a_Tmp[t.z].m_fPos - a_Tmp[t.x].m_fPos, a_Tmp[t.y].m_fPos - a_Tmp[t.x].m_fPos));
		a_TriData2[N].setData(a_Tmp[t.x].m_fPos, a_Tmp[t.y].m_fPos, a_Tmp[t.z].m_fPos,
			a_Tmp[t.x].m_fNormal, a_Tmp[t.y].m_fNormal, a_Tmp[t.z].m_fNormal);
#else

#endif
	}
}

void AnimatedMesh::launchKernels(void* a_DeviceTmp, AnimatedVertex* A, float4x4* m0, float4x4* m1, float a_lerp, uint3* triData, TriangleData* triData2)
{
	g_ComputeVertices << <k_Data.m_uVertexCount / 256 + 1, 256 >> >((e_KernelAnimatedMesh::e_TmpVertex*)a_DeviceTmp, A, m0, m1, a_lerp, k_Data.m_uVertexCount);
	ThrowCudaErrors(cudaDeviceSynchronize());
	g_ComputeTriangles << <m_sTriInfo.getLength() / 256 + 1, 256 >> >((e_KernelAnimatedMesh::e_TmpVertex*)a_DeviceTmp, triData, triData2, m_sTriInfo.getLength());
	ThrowCudaErrors(cudaDeviceSynchronize());
}

}
