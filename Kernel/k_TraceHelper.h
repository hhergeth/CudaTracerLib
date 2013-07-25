#pragma once
#include "..\Engine\e_Camera.h"
#include "..\Engine\e_DynamicScene.h"

//TODO : EXP scene doenst work with pppm
#define USE_EXP_SCENE

texture<float4, 1> t_nodesA;
texture<float4, 1> t_tris;
texture<int,  1>   t_triIndices;
texture<float4, 1> t_SceneNodes;
texture<float4, 1> t_NodeTransforms;
texture<float4, 1> t_NodeInvTransforms;

template<typename T> CUDA_ONLY_FUNC void swapDevice(T& a, T& b)
{
	T q = a;
	a = b;
	b = q;
}

enum
{
    MaxBlockHeight      = 6,            // Upper bound for blockDim.y.
    EntrypointSentinel  = 0x76543210,   // Bottom-most stack entry, indicating the end of traversal.
};

CUDA_ALIGN(16) __constant__ e_KernelDynamicScene g_SceneData;
CUDA_ALIGN(16) __device__ unsigned int g_NextRayCounter;
CUDA_ALIGN(16) __constant__ e_CameraData g_CameraData;
CUDA_ALIGN(16) __constant__ k_TracerRNGBuffer g_RNGData;

template<bool USE_ALPHA> inline __device__ bool k_TraceRayNode(const float3& dir, const float3& ori, TraceResult* a_Result, const e_Node* N)
{
	unsigned int mIndex = N->m_uMeshIndex;
	e_KernelMesh mesh = g_SceneData.m_sMeshData[mIndex];
	bool found = false;
	int traversalStack[64];
	traversalStack[0] = EntrypointSentinel;
	float   dirx = dir.x;
	float   diry = dir.y;
	float   dirz = dir.z;
	const float ooeps = exp2f(-80.0f);
	float   idirx = 1.0f / (fabsf(dir.x) > ooeps ? dir.x : copysignf(ooeps, dir.x));
	float   idiry = 1.0f / (fabsf(dir.y) > ooeps ? dir.y : copysignf(ooeps, dir.y));
	float   idirz = 1.0f / (fabsf(dir.z) > ooeps ? dir.z : copysignf(ooeps, dir.z));
	float   origx = ori.x;
	float	origy = ori.y;
	float	origz = ori.z;						// Ray origin.
	float   oodx = origx * idirx;
	float   oody = origy * idiry;
	float   oodz = origz * idirz;
	char*   stackPtr;                       // Current position in traversal stack.
	int     leafAddr;                       // First postponed leaf, non-negative if none.
	int     nodeAddr = EntrypointSentinel;  // Non-negative: current internal node, negative: second postponed leaf.
			stackPtr = (char*)&traversalStack[0];
			leafAddr = 0;   // No postponed leaf.
			nodeAddr = 0;   // Start from the root.
	while(nodeAddr != EntrypointSentinel)
	{
		while (unsigned int(nodeAddr) < unsigned int(EntrypointSentinel))
		{
			const float4 n0xy = tex1Dfetch(t_nodesA, mesh.m_uBVHNodeOffset + nodeAddr + 0); // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
			const float4 n1xy = tex1Dfetch(t_nodesA, mesh.m_uBVHNodeOffset + nodeAddr + 1); // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
			const float4 nz   = tex1Dfetch(t_nodesA, mesh.m_uBVHNodeOffset + nodeAddr + 2); // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
				  float4 tmp  = tex1Dfetch(t_nodesA, mesh.m_uBVHNodeOffset + nodeAddr + 3); // child_index0, child_index1
				  int2  cnodes= *(int2*)&tmp;
			const float c0lox = n0xy.x * idirx - oodx;
			const float c0hix = n0xy.y * idirx - oodx;
			const float c0loy = n0xy.z * idiry - oody;
			const float c0hiy = n0xy.w * idiry - oody;
			const float c0loz = nz.x   * idirz - oodz;
			const float c0hiz = nz.y   * idirz - oodz;
			const float c1loz = nz.z   * idirz - oodz;
			const float c1hiz = nz.w   * idirz - oodz;
			const float c0min = spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, 0);
			const float c0max = spanEndKepler  (c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, a_Result->m_fDist);
			const float c1lox = n1xy.x * idirx - oodx;
			const float c1hix = n1xy.y * idirx - oodx;
			const float c1loy = n1xy.z * idiry - oody;
			const float c1hiy = n1xy.w * idiry - oody;
			const float c1min = spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, 0);
			const float c1max = spanEndKepler  (c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, a_Result->m_fDist);
			bool swp = (c1min < c0min);
			bool traverseChild0 = (c0max >= c0min);
			bool traverseChild1 = (c1max >= c1min);
			if (!traverseChild0 && !traverseChild1)
			{
				nodeAddr = *(int*)stackPtr;
				stackPtr -= 4;
			}
			else
			{
				nodeAddr = (traverseChild0) ? cnodes.x : cnodes.y;
				if (traverseChild0 && traverseChild1)
				{
					if (swp)
						swapDevice(nodeAddr, cnodes.y);
					stackPtr += 4;
					*(int*)stackPtr = cnodes.y;
				}
			}

			if (nodeAddr < 0 && leafAddr  >= 0)     // Postpone max 1
			{
				leafAddr = nodeAddr;
				nodeAddr = *(int*)stackPtr;
				stackPtr -= 4;
			}

			unsigned int mask;
			asm("{\n"
				"   .reg .pred p;               \n"
				"setp.ge.s32        p, %1, 0;   \n"
				"vote.ballot.b32    %0,p;       \n"
				"}"
				: "=r"(mask)
				: "r"(leafAddr));
			if(!mask)
				break;
		}
		while (leafAddr < 0)
		{
			for (int triAddr = ~leafAddr;; triAddr += 3)
			{
				const float4 v00 = tex1Dfetch(t_tris, mesh.m_uBVHTriangleOffset + triAddr + 0);
				const float4 v11 = tex1Dfetch(t_tris, mesh.m_uBVHTriangleOffset + triAddr + 1);
				const float4 v22 = tex1Dfetch(t_tris, mesh.m_uBVHTriangleOffset + triAddr + 2);
				if (__float_as_int(v00.x) == 0x80000000)
					break;
				float Oz = v00.w - origx*v00.x - origy*v00.y - origz*v00.z;
				float invDz = 1.0f / (dirx*v00.x + diry*v00.y + dirz*v00.z);
				float t = Oz * invDz;
				if (t > 0.1f && t < a_Result->m_fDist)
				{
					float Ox = v11.w + origx*v11.x + origy*v11.y + origz*v11.z;
					float Dx = dirx*v11.x + diry*v11.y + dirz*v11.z;
					float u = Ox + t*Dx;
					if (u >= 0.0f)
					{
						float Oy = v22.w + origx*v22.x + origy*v22.y + origz*v22.z;
						float Dy = dirx*v22.x + diry*v22.y + dirz*v22.z;
						float v = Oy + t*Dy;
						if (v >= 0.0f && u + v <= 1.0f)
						{
							unsigned int ti = tex1Dfetch(t_triIndices, triAddr + mesh.m_uBVHIndicesOffset);
							e_TriangleData* tri = g_SceneData.m_sTriData.Data + ti + mesh.m_uTriangleOffset;
							int q = 1;
							if(q)
							{
								a_Result->m_pNode = N;
								a_Result->m_pTri = tri;
								a_Result->m_fUV = make_float2(u, v);
								a_Result->m_fDist = t;
								found = true;
							}
						}
					}
				}
			}
			leafAddr = nodeAddr;
			if (nodeAddr < 0)
			{
				nodeAddr = *(int*)stackPtr;
				stackPtr -= 4;
			}
		}
	}
	return found;
}

//a_Result->m_fDist = length(modl.TransformNormal(dir*a_Result->m_fDist));
//a_Result->m_fDist = length(modl2.TransformNormal(d*a_Result->m_fDist));

#ifdef USE_EXP_SCENE
template<bool USE_ALPHA> inline __device__ bool k_TraceRay(const float3& dir, const float3& ori, TraceResult* a_Result)
{
	if(!g_SceneData.m_sNodeData.UsedCount)
		return false;
	int traversalStackOuter[64];
	int at = 1;
	traversalStackOuter[0] = g_SceneData.m_sSceneBVH.m_sStartNode;
	const float ooeps = exp2f(-80.0f);
	float3 O, I;
	I.x = 1.0f / (fabsf(dir.x) > ooeps ? dir.x : copysignf(ooeps, dir.x));
	I.y = 1.0f / (fabsf(dir.y) > ooeps ? dir.y : copysignf(ooeps, dir.y));
	I.z = 1.0f / (fabsf(dir.z) > ooeps ? dir.z : copysignf(ooeps, dir.z));
	O = I * ori;
	while(at)
	{
		int nodeAddrOuter = traversalStackOuter[--at];
		while (nodeAddrOuter >= 0)
		{
			const float4 n0xy = tex1Dfetch(t_SceneNodes, nodeAddrOuter * 4 + 0); // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
			const float4 n1xy = tex1Dfetch(t_SceneNodes, nodeAddrOuter * 4 + 1); // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
			const float4 nz   = tex1Dfetch(t_SceneNodes, nodeAddrOuter * 4 + 2); // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
				  float4 tmp  = tex1Dfetch(t_SceneNodes, nodeAddrOuter * 4 + 3); // child_index0, child_index1
			int2  cnodesOuter = *(int2*)&tmp;
			const float c0lox = n0xy.x * I.x - O.x;
			const float c0hix = n0xy.y * I.x - O.x;
			const float c0loy = n0xy.z * I.y - O.y;
			const float c0hiy = n0xy.w * I.y - O.y;
			const float c0loz = nz.x   * I.z - O.z;
			const float c0hiz = nz.y   * I.z - O.z;
			const float c1loz = nz.z   * I.z - O.z;
			const float c1hiz = nz.w   * I.z - O.z;
			const float c0min = spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, 0);
			const float c0max = spanEndKepler  (c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, a_Result->m_fDist);
			const float c1lox = n1xy.x * I.x - O.x;
			const float c1hix = n1xy.y * I.x - O.x;
			const float c1loy = n1xy.z * I.y - O.y;
			const float c1hiy = n1xy.w * I.y - O.y;
			const float c1min = spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, 0);
			const float c1max = spanEndKepler  (c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, a_Result->m_fDist);
			bool swpOuter = (c1min < c0min);
			bool traverseChild0Outer = (c0max >= c0min);
			bool traverseChild1Outer = (c1max >= c1min);
			if ((!traverseChild0Outer && !traverseChild1Outer) && at)
				nodeAddrOuter = traversalStackOuter[--at];
			else if(!traverseChild0Outer && !traverseChild1Outer)
			{//empty stack and nowhere to go...
				nodeAddrOuter = 0;
				break;
			}
			else
			{
				nodeAddrOuter = (traverseChild0Outer) ? cnodesOuter.x : cnodesOuter.y;
				if (traverseChild0Outer && traverseChild1Outer)
				{
					if (swpOuter)
						swapDevice(nodeAddrOuter, cnodesOuter.y);
					traversalStackOuter[at++] = cnodesOuter.y;
				}
			}
		}
		if(nodeAddrOuter < 0)
		{
			int node = ~nodeAddrOuter;
			e_Node* N = g_SceneData.m_sNodeData.Data + node;
			//transform a_Result->m_fDist to local system
			float4x4 modl = N->getInvWorldMatrix();
			float3 d = modl.TransformNormal(dir), o = modl.TransformNormal(ori) + modl.Translation();
			float3 scale = modl.Scale();
			float scalef = length(d / scale);
			a_Result->m_fDist /= scalef;
			k_TraceRayNode<USE_ALPHA>(d, o, a_Result, N);
			//transform a_Result->m_fDist back to world
			a_Result->m_fDist *= scalef;
		}
	}
	return a_Result->hasHit();
}
#else
template<bool USE_ALPHA> inline __device__ bool k_TraceRay(const float3& dir, const float3& ori, TraceResult* a_Result)
{
	e_Node* N = g_SceneData.m_sNodeData.Data;
	return k_TraceRayNode<USE_ALPHA>(dir, ori, a_Result, N);
}
#endif

CUDA_ONLY_FUNC TraceResult k_TraceRay(const Ray& r)
{
	TraceResult r2;
	r2.Init();
	k_TraceRay<false>(r.direction, r.origin, &r2);
	return r2;
}

CUDA_FUNC_IN float3 Le(const float3& p, const float3& n, const float3& w, TraceResult& r2, e_KernelDynamicScene& scene) 
{
	unsigned int i = scene.m_sMatData[r2.m_pTri->getMatIndex(r2.m_pNode->m_uMaterialOffset)].NodeLightIndex;
	if(i == -1)
		return make_float3(0);
	unsigned int j = r2.m_pNode->m_uLightIndices[i];
	if(j == -1)
		return make_float3(0);
	e_KernelLight& l = scene.m_sLightData[j];
	return dot(w, n) > 0 ? l.L(p, n, w) : make_float3(0);
}

CUDA_FUNC_IN unsigned int LightIndex(TraceResult& r2, e_KernelDynamicScene& scene)
{
	unsigned int i = scene.m_sMatData[r2.m_pTri->getMatIndex(r2.m_pNode->m_uMaterialOffset)].NodeLightIndex;
	if(i == -1)
		return -1;
	unsigned int j = r2.m_pNode->m_uLightIndices[i];
	return j;
}

Onb TraceResult::lerpOnb()
{
	return m_pTri->lerpOnb(m_fUV, m_pNode->getWorldMatrix(), 0);
}

unsigned int TraceResult::getMatIndex()
{
	return m_pTri->getMatIndex(m_pNode->m_uMaterialOffset);
}

float2 TraceResult::lerpUV()
{
	return m_pTri->lerpUV(m_fUV);
}

void TraceResult::GetBSDF(const float3& p, const e_KernelMaterial* a_Mats, e_KernelBSDF* bsdf)
{
	m_pTri->GetBSDF(p, m_fUV, m_pNode->getWorldMatrix(), a_Mats, m_pNode->m_uMaterialOffset, bsdf);
}

e_KernelBSDF TraceResult::GetBSDF(const float3& p, const e_KernelMaterial* a_Mats)
{
	e_KernelBSDF bs;
	GetBSDF(p, a_Mats, &bs);
	return bs;
}

bool TraceResult::GetBSSRDF(const float3& p, const e_KernelMaterial* a_Mats, e_KernelBSSRDF* bssrdf)
{
	return m_pTri->GetBSSRDF(p, m_fUV, m_pNode->getWorldMatrix(), a_Mats, m_pNode->m_uMaterialOffset, bssrdf);
}

//do not!!! use a method here, the compiler will fuck up the textures.
#define k_INITIALIZE(a_Data) \
	{ \
		size_t offset; \
		cudaChannelFormatDesc cd0 = cudaCreateChannelDesc<float4>(), cd1 = cudaCreateChannelDesc<int>(); \
		cudaError_t \
		r = cudaBindTexture(&offset, &t_nodesA, a_Data.m_sBVHNodeData.Data, &cd0, a_Data.m_sBVHNodeData.UsedCount * sizeof(e_BVHNodeData)); \
		r = cudaBindTexture(&offset, &t_tris, a_Data.m_sBVHIntData.Data, &cd0, a_Data.m_sBVHIntData.UsedCount * sizeof(e_TriIntersectorData)); \
		r = cudaBindTexture(&offset, &t_triIndices, a_Data.m_sBVHIndexData.Data, &cd1, a_Data.m_sBVHIndexData.UsedCount * sizeof(int)); \
		r = cudaBindTexture(&offset, &t_SceneNodes, a_Data.m_sSceneBVH.m_pNodes, &cd0, a_Data.m_sBVHNodeData.UsedCount * sizeof(e_BVHNodeData)); \
		r = cudaBindTexture(&offset, &t_NodeTransforms, a_Data.m_sSceneBVH.m_pNodeTransforms, &cd0, a_Data.m_sNodeData.UsedCount * sizeof(float4x4)); \
		r = cudaBindTexture(&offset, &t_NodeInvTransforms, a_Data.m_sSceneBVH.m_pInvNodeTransforms, &cd0, a_Data.m_sNodeData.UsedCount * sizeof(float4x4)); \
	}

#define k_STARTPASS(a_Scene, a_Camera, a_RngBuf) \
	{ \
		unsigned int b = 0; \
		cudaMemcpyToSymbol(g_NextRayCounter, &b, sizeof(unsigned int)); \
		e_CameraData d; \
		a_Camera->getData(d); \
		e_KernelDynamicScene d2 = a_Scene->getKernelSceneData(); \
		cudaMemcpyToSymbol(g_SceneData, &d2, sizeof(e_KernelDynamicScene)); \
		cudaMemcpyToSymbol(g_CameraData, &d, sizeof(d)); \
		cudaMemcpyToSymbol(g_RNGData, &a_RngBuf, sizeof(k_TracerRNGBuffer)); \
	}