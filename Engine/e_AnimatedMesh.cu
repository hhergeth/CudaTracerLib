#include "e_AnimatedMesh.h"
#include "e_DynamicScene.h"

//low...high
//x=(0,1,2,3), y=(4,5,6,7)
#define SELECT(x,y,o) o < 4 ? (x >> (o * 8)) : (x >> (o * 8 - 32))
#define SCRAMBLE(x,y,a,b,c,d) unsigned int(((SELECT(x, y, d)) << 24) | ((SELECT(x, y, c) << 16)) | ((SELECT(x, y, b) << 8)) | ((SELECT(x, y, a))))

CUDA_ONLY_FUNC float4x4 d_Compute(float4x4* a_Matrices, e_AnimatedVertex& v)
{
	float4x4 mat(0,0,0,0,
				0,0,0,0,
				0,0,0,0,
				0,0,0,0);
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
		a_Dest[N].m_fNormal = -normalize(lerp(n0, n1, a_Lerp));

		float3 t0 = mat0.TransformNormal(v.m_fTangent), t1 = mat1.TransformNormal(v.m_fTangent);
		a_Dest[N].m_fTangent = normalize(lerp(t0, t1, a_Lerp));

		float3 b0 = mat0.TransformNormal(v.m_fBitangent), b1 = mat1.TransformNormal(v.m_fBitangent);
		a_Dest[N].m_fBiTangent = normalize(lerp(b0, b1, a_Lerp));
	}
}

CUDA_FUNC_IN unsigned int cnv(float3& f, unsigned int off = 0)
{
	uchar2 r = NormalizedFloat3ToUchar2(f);
	return ((unsigned int)r.x | ((unsigned int)r.y << 8)) << off;
}

__global__ void g_ComputeTriangles(e_TmpVertex* a_Tmp, uint3* a_TriData, e_TriangleData* a_TriData2, unsigned int a_TCount)
{
	unsigned int N = blockIdx.x * blockDim.x + threadIdx.x;
	if(N < a_TCount)
	{
		uint3 t = a_TriData[N];
		float3 n = normalize(cross(a_Tmp[t.y].m_fPos - a_Tmp[t.x].m_fPos, a_Tmp[t.z].m_fPos - a_Tmp[t.x].m_fPos));
		a_TriData2[N].m_sDeviceData.Row0.x = cnv(a_Tmp[t.x].m_fNormal) | cnv(a_Tmp[t.y].m_fNormal, 16);
		a_TriData2[N].m_sDeviceData.Row0.y = cnv(a_Tmp[t.z].m_fNormal) | cnv(a_Tmp[t.x].m_fTangent, 16);
		a_TriData2[N].m_sDeviceData.Row0.z = cnv(a_Tmp[t.y].m_fTangent) | cnv(a_Tmp[t.z].m_fTangent, 16);
#define TOUINT3(v) ((unsigned char((v.x + 1) * 127.0f) << 16) | (unsigned char((v.y + 1) * 127.0f) << 8) | (unsigned char((v.z + 1) * 127.0f)))
		//unsigned int n0 = TOUINT3(a_Tmp[t.x].m_fNormal), n1 = TOUINT3(a_Tmp[t.y].m_fNormal), n2 = TOUINT3(a_Tmp[t.z].m_fNormal);
		//unsigned int t0 = TOUINT3(a_Tmp[t.x].m_fTangent), t1 = TOUINT3(a_Tmp[t.y].m_fTangent), t2 = TOUINT3(a_Tmp[t.z].m_fTangent);
		//unsigned int b0 = TOUINT3(a_Tmp[t.x].m_fBiTangent), b1 = TOUINT3(a_Tmp[t.y].m_fBiTangent), b2 = TOUINT3(a_Tmp[t.z].m_fBiTangent);
		//a_TriData2[N].m_sDeviceData.Row0 = make_uint4(n1 << 24 | n0, n2 << 16 | n1 >> 8, t0 << 8 | n2 >> 16, t2 << 24 | t1);
		//a_TriData2[N].m_sDeviceData.Row1 = make_uint3(b0 << 8 | t2 >> 8, b1 << 8 | b0 >> 16, a_TriData2[N].getMatIndex(0) << 24 | b2);
		//a_TriData2[N].NOR[0] = a_Tmp[t.x].m_fNormal; a_TriData2[N].NOR[1] = a_Tmp[t.y].m_fNormal; a_TriData2[N].NOR[2] = a_Tmp[t.z].m_fNormal;
#undef TOUINT3
	}
}

CUDA_DEVICE AABB g_BOX;

__global__ void g_ComputeBVHState(e_TriIntersectorData* a_BVHIntersectionData, e_BVHNodeData* a_BVHNodeData, int* a_BVHIndexData, e_TmpVertex* a_Tmp, uint3* a_TriData, e_BVHLevelEntry* a_Level, unsigned int a_NCount)
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
		if(N == 0 && a_NCount == 1)
			g_BOX = box;
	}
}

void e_AnimatedMesh::k_ComputeState(unsigned int a_Anim, unsigned int a_Frame, float a_Lerp, e_KernelDynamicScene a_Data, e_Stream<e_BVHNodeData>* a_BVHNodeStream, e_TmpVertex* a_DeviceTmp)
{
	unsigned int n = (a_Frame + 1) % m_pAnimations[a_Anim].m_uNumFrames;
	float4x4* m0 = TRANS(m_pAnimData + m_pAnimations[a_Anim].m_uDataOffset + k_Data.m_uJointCount * a_Frame), *m1 = TRANS(m_pAnimData + m_pAnimations[a_Anim].m_uDataOffset + k_Data.m_uJointCount * n);
	g_ComputeVertices<<<k_Data.m_uVertexCount / 256 + 1, 256>>>(a_DeviceTmp, TRANS(m_pVertices), m0, m1, a_Lerp, k_Data.m_uVertexCount);
	cudaThreadSynchronize();
	g_ComputeTriangles<<<k_Data.m_uTriangleCount / 256 + 1, 256>>>(a_DeviceTmp, TRANS(m_pTriangles), a_Data.m_sTriData.Data + m_sTriInfo.getIndex(), k_Data.m_uTriangleCount);
	cudaThreadSynchronize();
	for(int l = k_Data.m_uBVHLevelCount - 1; l >= 0; l--){
		g_ComputeBVHState<<<m_pLevels[l].y / 256 + 1, 256>>>(a_Data.m_sBVHIntData.Data + m_sIntInfo.getIndex(), a_Data.m_sBVHNodeData.Data + m_sNodeInfo.getIndex(), a_Data.m_sBVHIndexData.Data + m_sIndicesInfo.getIndex(), a_DeviceTmp, TRANS(m_pTriangles), TRANS(m_pLevelEntries + m_pLevels[l].x), m_pLevels[l].y);
		cudaThreadSynchronize();}
	cudaMemcpyFromSymbol(&m_sLocalBox, g_BOX, sizeof(AABB));
}