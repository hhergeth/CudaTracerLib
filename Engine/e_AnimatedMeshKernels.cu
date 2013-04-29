#include "e_AnimatedMesh.h"
#include "e_DynamicScene.h"

__device__ float4x4 d_Compute(float4x4* a_Matrices, e_AnimatedVertex& v)
{
	float4x4 mat(0,0,0,0,
				0,0,0,0,
				0,0,0,0,
				0,0,0,0);
	for(int i = 0; i < g_uMaxWeights; i++)
	{
		if(v.m_fBoneWeights[i] == 0)
			break;
		mat = mat + a_Matrices[v.m_cBoneIndices[i]] * (float(v.m_fBoneWeights[i]) / 255.0f);
	}
	return mat;
}

__global__ void g_ComputeVertices(e_TmpVertex* a_Dest, e_AnimatedVertex* a_Source, float4x4* a_Matrices, float4x4* a_Matrices2, float a_Lerp, unsigned int a_VCount)
{
	unsigned int N = blockIdx.x * blockDim.x + threadIdx.x;
	if(N < a_VCount)
	{
		e_AnimatedVertex v = a_Source[N];
		float4x4 mat0 = d_Compute(a_Matrices, v);
		float4x4 mat1 = d_Compute(a_Matrices2, v);
		float3 v0 = mat0 * v.m_fVertexPos, v1 = mat1 * v.m_fVertexPos;
		a_Dest[N].m_fPos = lerp(v0, v1, a_Lerp);
		float3 n0 = mat0.TransformNormal(v.m_fNormal), n1 = mat1.TransformNormal(v.m_fNormal);
		a_Dest[N].m_fNormal = normalize(lerp(n0, n1, a_Lerp));
	}
}

__global__ void g_ComputeTriangles(e_TmpVertex* a_Tmp, uint3* a_TriData, e_TriangleData* a_TriData2, unsigned int a_TCount)
{
	unsigned int N = blockIdx.x * blockDim.x + threadIdx.x;
	if(N < a_TCount)
	{/*
		uint3 t = a_TriData[N];
		a_TriData2[N].setVertexNormal(a_Tmp[t.x].m_fNormal, 0);
		a_TriData2[N].setVertexNormal(a_Tmp[t.y].m_fNormal, 1);
		a_TriData2[N].setVertexNormal(a_Tmp[t.z].m_fNormal, 2);

		float3 v0 = a_Tmp[t.x].m_fPos, v1 = a_Tmp[t.y].m_fPos, v2 = a_Tmp[t.z].m_fPos;
		float3 n = normalize(cross(v0 - v2, v1 - v2));
		a_TriData2[N].setNormal(n);*/
	}
}

__global__ void g_ComputeBVHState(e_TriIntersectorData* a_BVHIntersectionData, e_BVHNodeData* a_BVHNodeData, int* a_BVHIndexData, e_TmpVertex* a_Tmp, uint3* a_TriData, e_BVHLevelEntry* a_Level, unsigned int a_NCount)
{
	unsigned int N = blockIdx.x * blockDim.x + threadIdx.x;
	if(N < a_NCount)
	{
		e_BVHLevelEntry e = a_Level[N];
		AABB box;
		box = box.Identity();
		if(e.m_sNode > 0)
		{
			AABB l, r;
			a_BVHNodeData[e.m_sNode / 4].getBox(l, r);
			box.minV = fminf(l.minV, r.minV);
			box.maxV = fmaxf(l.maxV, r.maxV);
		}
		else if(e.m_sNode < 0)
		{
			for (int triAddr = ~e.m_sNode;; triAddr += 3)
			{
				int i = a_BVHIndexData[triAddr];
				if(i == -1)
					break;
				uint3 t = a_TriData[i];
				float3 v0 = a_Tmp[t.x].m_fPos, v1 = a_Tmp[t.y].m_fPos, v2 = a_Tmp[t.z].m_fPos;
				box.Enlarge(v0);
				box.Enlarge(v1);
				box.Enlarge(v2);
				e_TriIntersectorData* T = (e_TriIntersectorData*)((float4*)a_BVHIntersectionData + triAddr);
				T->setData(v0, v1, v2);
			}
		}
		if(e.m_sSide == -1)
			a_BVHNodeData[e.m_sParent / 4].setLeft(box);
		else if(e.m_sSide == 1)
			a_BVHNodeData[e.m_sParent / 4].setRight(box);
	}
}

void e_AnimatedMesh::k_ComputeState(unsigned int a_Anim, unsigned int a_Frame, float a_Lerp, e_KernelDynamicScene a_Data, e_DataStream<e_BVHNodeData>* a_BVHNodeStream, e_TmpVertex* a_DeviceTmp)
{
	unsigned int n = (a_Frame + 1) % m_pAnimations[a_Anim].m_uNumFrames;
	float4x4* m0 = TRANS(m_pAnimData + m_pAnimations[a_Anim].m_uDataOffset + k_Data.m_uJointCount * a_Frame), *m1 = TRANS(m_pAnimData + m_pAnimations[a_Anim].m_uDataOffset + k_Data.m_uJointCount * n);
	g_ComputeVertices<<<k_Data.m_uVertexCount / 256 + 1, 256>>>(a_DeviceTmp, TRANS(m_pVertices), m0, m1, a_Lerp, k_Data.m_uVertexCount);
	SYNC_BAD_CUDA_CALLD
	g_ComputeTriangles<<<k_Data.m_uTriangleCount / 256 + 1, 256>>>(a_DeviceTmp, TRANS(m_pTriangles), a_Data.m_sTriData.Data + m_sTriInfo.getIndex(), k_Data.m_uTriangleCount);
	SYNC_BAD_CUDA_CALLD
	for(int l = k_Data.m_uBVHLevelCount - 1; l > 0; l--)
		g_ComputeBVHState<<<m_pLevels[l].y / 256 + 1, 256>>>(a_Data.m_sBVHIntData.Data + m_sIntInfo.getIndex(), a_Data.m_sBVHNodeData.Data + m_sNodeInfo.getIndex(), a_Data.m_sBVHIndexData.Data + m_sIndicesInfo.getIndex(), a_DeviceTmp, TRANS(m_pTriangles), TRANS(m_pLevelEntries + m_pLevels[l].x), m_pLevels[l].y);
	SYNC_BAD_CUDA_CALLD

	a_BVHNodeStream->CopyDeviceToHost(m_sNodeInfo.getIndex(), m_sNodeInfo.getLength());
	AABB l, r;
	a_BVHNodeStream[0](m_sNodeInfo.getIndex())->getBox(l, r);
	l.Enlarge(r);
	this->m_sLocalBox = l;
}