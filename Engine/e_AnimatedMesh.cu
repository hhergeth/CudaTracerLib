#include "e_AnimatedMesh.h"
#include "e_KernelDynamicScene.h"

//low...high
//x=(0,1,2,3), y=(4,5,6,7)
#define SELECT(x,y,o) o < 4 ? (x >> (o * 8)) : (x >> (o * 8 - 32))
#define SCRAMBLE(x,y,a,b,c,d) unsigned int(((SELECT(x, y, d)) << 24) | ((SELECT(x, y, c) << 16)) | ((SELECT(x, y, b) << 8)) | ((SELECT(x, y, a))))

CUDA_ONLY_FUNC float4x4 d_Compute(float4x4* a_Matrices, e_AnimatedVertex& v)
{
	float4x4 mat;
	mat.zeros();
	for(int i = 0; i < g_uMaxWeights; i++)
	{
		if(v.m_fBoneWeights[i] == 0)
			break;
		float4x4 m = a_Matrices[v.m_cBoneIndices[i]];
		m = m * (float(v.m_fBoneWeights[i]) / 255.0f);
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
		a_Dest[N].m_fNormal = -normalize(math::lerp(n0, n1, a_lerp));

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
		a_TriData2[N].setData(a_Tmp[t.x].m_fPos, a_Tmp[t.y].m_fPos, a_Tmp[t.z].m_fPos,
							  a_Tmp[t.x].m_fNormal, a_Tmp[t.y].m_fNormal, a_Tmp[t.z].m_fNormal);
#else
		//a_TriData2[N].m_sDeviceData.Row0.
#endif
	}
}

CUDA_DEVICE AABB g_BOX;

__global__ void g_ComputeBVHState(e_TriIntersectorData* a_BVHIntersectionData, e_BVHNodeData* a_BVHNodeData, e_TriIntersectorData2* a_BVHIntersectionData2, e_TmpVertex* a_Tmp, uint3* a_TriData, e_BVHLevelEntry* a_Level, unsigned int a_NCount)
{
	unsigned int N = blockIdx.x * blockDim.x + threadIdx.x;
	if(N < a_NCount)
	{
		e_BVHLevelEntry e = a_Level[N];
		AABB box;
		box = box.Identity();
		if(e.m_sNode >= 0)
		{
			AABB l, r;
			a_BVHNodeData[e.m_sNode].getBox(l, r);
			box.minV = min(l.minV, r.minV);
			box.maxV = max(l.maxV, r.maxV);
		}
		else if(e.m_sNode < 0)
		{
			for (int triAddr = ~e.m_sNode; ; triAddr++)
			{
				e_TriIntersectorData2 i = a_BVHIntersectionData2[triAddr];
				uint3 t = a_TriData[i.getIndex()];
				Vec3f v0 = a_Tmp[t.x].m_fPos, v1 = a_Tmp[t.y].m_fPos, v2 = a_Tmp[t.z].m_fPos;
				box.Enlarge(v0);
				box.Enlarge(v1);
				box.Enlarge(v2);
				a_BVHIntersectionData[triAddr].setData(v0, v1, v2);//no change in indices
				if(i.getFlag())
					break;
			}
		}
		if(e.m_sSide == 1)
			a_BVHNodeData[e.m_sParent].setLeft(box);
		else if(e.m_sSide == -1)
			a_BVHNodeData[e.m_sParent].setRight(box);
		if(N == 0 && a_NCount == 1)
			g_BOX = box;
	}
}

void e_AnimatedMesh::k_ComputeState(unsigned int a_Anim, unsigned int a_Frame, float a_lerp, e_KernelDynamicScene a_Data, e_Stream<e_BVHNodeData>* a_BVHNodeStream, e_TmpVertex* a_DeviceTmp)
{
	unsigned int n = (a_Frame + 1) % m_pAnimations[a_Anim].m_pFrames.size();
	float4x4* m0 = (float4x4*)m_pAnimations[a_Anim].m_pFrames[a_Frame].m_sMatrices.getDevice();
	float4x4* m1 = (float4x4*)m_pAnimations[a_Anim].m_pFrames[n].m_sMatrices.getDevice();
	g_ComputeVertices<<<k_Data.m_uVertexCount / 256 + 1, 256>>>(a_DeviceTmp, (e_AnimatedVertex*)m_sVertices.getDevice(), m0, m1, a_lerp, k_Data.m_uVertexCount);
	cudaThreadSynchronize();
	ThrowCudaErrors();
	g_ComputeTriangles<<<m_sTriInfo.getLength() / 256 + 1, 256>>>(a_DeviceTmp, (uint3*)m_sTriangles.getDevice(), m_sTriInfo.getDevice(), m_sTriInfo.getLength());
	cudaThreadSynchronize();
	ThrowCudaErrors();
	for(int l = m_sHierchary.m_uNumLevels - 1; l >= 0; l--)
	{
		int l2 = m_sHierchary.numInLevel(l);
		g_ComputeBVHState<<<l2 / 256 + 1, 256>>>(m_sIntInfo.getDevice(), m_sNodeInfo.getDevice(), m_sIndicesInfo.getDevice(), a_DeviceTmp, 
												 (uint3*)m_sTriangles.getDevice(), m_sHierchary.getLevelStartOnDevice(l), l2);
		cudaThreadSynchronize();
		ThrowCudaErrors();
	}
	cudaMemcpyFromSymbol(&m_sLocalBox, g_BOX, sizeof(AABB));
}