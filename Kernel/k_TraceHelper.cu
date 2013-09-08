#include "k_TraceHelper.h"

e_KernelDynamicScene g_SceneDataDevice;
unsigned int g_RayTracedCounterDevice;
e_CameraData g_CameraDataDevice;
k_TracerRNGBuffer g_RNGDataDevice;

e_KernelDynamicScene g_SceneDataHost;
volatile LONG g_RayTracedCounterHost;
e_CameraData g_CameraDataHost;
k_TracerRNGBuffer g_RNGDataHost;

texture<float4, 1> t_nodesA;
texture<float4, 1> t_tris;
texture<int,  1>   t_triIndices;
texture<float4, 1> t_SceneNodes;
texture<float4, 1> t_NodeTransforms;
texture<float4, 1> t_NodeInvTransforms;

bool k_TraceRayNode(const float3& dir, const float3& ori, TraceResult* a_Result, const e_Node* N, int lastIndex)
{
	const bool USE_ALPHA = true;
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
#ifdef ISCUDA
			const float4 n0xy = tex1Dfetch(t_nodesA, mesh.m_uBVHNodeOffset + nodeAddr + 0); // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
			const float4 n1xy = tex1Dfetch(t_nodesA, mesh.m_uBVHNodeOffset + nodeAddr + 1); // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
			const float4 nz   = tex1Dfetch(t_nodesA, mesh.m_uBVHNodeOffset + nodeAddr + 2); // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
				  float4 tmp  = tex1Dfetch(t_nodesA, mesh.m_uBVHNodeOffset + nodeAddr + 3); // child_index0, child_index1
#else
			float4* dat = (float4*)g_SceneData.m_sBVHNodeData.Data;
			const float4 n0xy = dat[mesh.m_uBVHNodeOffset + nodeAddr + 0];
			const float4 n1xy = dat[mesh.m_uBVHNodeOffset + nodeAddr + 1];
			const float4 nz   = dat[mesh.m_uBVHNodeOffset + nodeAddr + 2];
				  float4 tmp  = dat[mesh.m_uBVHNodeOffset + nodeAddr + 3];
#endif
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
						swapk(&nodeAddr, &cnodes.y);
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
#ifdef ISCUDA
			asm("{\n"
				"   .reg .pred p;               \n"
				"setp.ge.s32        p, %1, 0;   \n"
				"vote.ballot.b32    %0,p;       \n"
				"}"
				: "=r"(mask)
				: "r"(leafAddr));
#else
			mask = leafAddr >= 0;
#endif
			if(!mask)
				break;
		}
		while (leafAddr < 0)
		{
			for (int triAddr = ~leafAddr;; triAddr += 3)
			{
#ifdef ISCUDA
				const float4 v00 = tex1Dfetch(t_tris, mesh.m_uBVHTriangleOffset + triAddr + 0);
				const float4 v11 = tex1Dfetch(t_tris, mesh.m_uBVHTriangleOffset + triAddr + 1);
				const float4 v22 = tex1Dfetch(t_tris, mesh.m_uBVHTriangleOffset + triAddr + 2);
#else
				float4* dat = (float4*)g_SceneData.m_sBVHIntData.Data;
				const float4 v00 = dat[mesh.m_uBVHTriangleOffset + triAddr + 0];
				const float4 v11 = dat[mesh.m_uBVHTriangleOffset + triAddr + 1];
				const float4 v22 = dat[mesh.m_uBVHTriangleOffset + triAddr + 2];
#endif
				if (__float_as_int(v00.x) == 0x80000000)
					break;
				float Oz = v00.w - origx*v00.x - origy*v00.y - origz*v00.z;
				float invDz = 1.0f / (dirx*v00.x + diry*v00.y + dirz*v00.z);
				float t = Oz * invDz;
				if (t > 1e-3f && t < a_Result->m_fDist && triAddr != lastIndex)
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
#ifdef ISCUDA
							unsigned int ti = tex1Dfetch(t_triIndices, triAddr + mesh.m_uBVHIndicesOffset);
#else
							unsigned int ti = g_SceneData.m_sBVHIndexData.Data[triAddr + mesh.m_uBVHIndicesOffset];
#endif
							e_TriangleData* tri = g_SceneData.m_sTriData.Data + ti + mesh.m_uTriangleOffset;
							int q = 1;
							if(USE_ALPHA)
							{
								e_KernelMaterial* mat = g_SceneData.m_sMatData.Data + tri->getMatIndex(N->m_uMaterialOffset);
								float a = mat->SampleAlphaMap(MapParameters(make_float3(0), tri->lerpUV(make_float2(u,v)), Frame()));
								q = a >= mat->m_fAlphaThreshold;
							}
							if(q)
							{
								a_Result->__internal__earlyExit = triAddr;
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

bool k_TraceRay(const float3& dir, const float3& ori, TraceResult* a_Result)
{
	int lastIndex = a_Result->__internal__earlyExit;
	const e_Node* lastNode = a_Result->m_pNode;
#ifdef ISCUDA
	atomicInc(&g_RayTracedCounterDevice, 0xffffffff);
#else
	InterlockedIncrement(&g_RayTracedCounterHost);
#endif
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
#ifdef ISCUDA
			const float4 n0xy = tex1Dfetch(t_SceneNodes, nodeAddrOuter * 4 + 0); // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
			const float4 n1xy = tex1Dfetch(t_SceneNodes, nodeAddrOuter * 4 + 1); // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
			const float4 nz   = tex1Dfetch(t_SceneNodes, nodeAddrOuter * 4 + 2); // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
				  float4 tmp  = tex1Dfetch(t_SceneNodes, nodeAddrOuter * 4 + 3); // child_index0, child_index1
#else
			const float4 n0xy = g_SceneData.m_sSceneBVH.m_pNodes[nodeAddrOuter].a;
			const float4 n1xy = g_SceneData.m_sSceneBVH.m_pNodes[nodeAddrOuter].b;
			const float4 nz   = g_SceneData.m_sSceneBVH.m_pNodes[nodeAddrOuter].c;
				  float4 tmp  = g_SceneData.m_sSceneBVH.m_pNodes[nodeAddrOuter].d;
#endif
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
						swapk(&nodeAddrOuter, &cnodesOuter.y);
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
			k_TraceRayNode(d, o, a_Result, N, lastNode == N ? lastIndex : -1);
			//transform a_Result->m_fDist back to world
			a_Result->m_fDist *= scalef;
		}
	}
	return a_Result->hasHit();
}

void k_INITIALIZE(const e_KernelDynamicScene& a_Data)
{
	size_t offset;
	cudaChannelFormatDesc cd0 = cudaCreateChannelDesc<float4>(), cd1 = cudaCreateChannelDesc<int>();
	cudaError_t
	r = cudaBindTexture(&offset, &t_nodesA, a_Data.m_sBVHNodeData.Data, &cd0, a_Data.m_sBVHNodeData.UsedCount * sizeof(e_BVHNodeData));
	r = cudaBindTexture(&offset, &t_tris, a_Data.m_sBVHIntData.Data, &cd0, a_Data.m_sBVHIntData.UsedCount * sizeof(e_TriIntersectorData));
	r = cudaBindTexture(&offset, &t_triIndices, a_Data.m_sBVHIndexData.Data, &cd1, a_Data.m_sBVHIndexData.UsedCount * sizeof(int));
	r = cudaBindTexture(&offset, &t_SceneNodes, a_Data.m_sSceneBVH.m_pNodes, &cd0, a_Data.m_sBVHNodeData.UsedCount * sizeof(e_BVHNodeData));
	r = cudaBindTexture(&offset, &t_NodeTransforms, a_Data.m_sSceneBVH.m_pNodeTransforms, &cd0, a_Data.m_sNodeData.UsedCount * sizeof(float4x4));
	r = cudaBindTexture(&offset, &t_NodeInvTransforms, a_Data.m_sSceneBVH.m_pInvNodeTransforms, &cd0, a_Data.m_sNodeData.UsedCount * sizeof(float4x4));
}

void k_STARTPASS(e_DynamicScene* a_Scene, e_Camera* a_Camera, const k_TracerRNGBuffer& a_RngBuf)
{
	unsigned int b = 0;
	cudaMemcpyToSymbol(g_RayTracedCounterDevice, &b, sizeof(unsigned int));
	e_CameraData d = a_Camera->getData();
	e_KernelDynamicScene d2 = a_Scene->getKernelSceneData();
	cudaMemcpyToSymbol(g_SceneDataDevice, &d2, sizeof(e_KernelDynamicScene));
	cudaMemcpyToSymbol(g_CameraDataDevice, &d, sizeof(d));
	cudaMemcpyToSymbol(g_RNGDataDevice, &a_RngBuf, sizeof(k_TracerRNGBuffer));

	g_SceneDataHost = a_Scene->getKernelSceneData(false);
	g_CameraDataHost = d;
	g_RNGDataHost = a_RngBuf;
	g_RayTracedCounterHost = 0;
}

Spectrum TraceResult::Le(const float3& p, const float3& n, const float3& w) const 
{
	unsigned int i = LightIndex();
	if(i == 0xffffffff)
		return Spectrum(0.0f);
	else return g_SceneData.m_sLightData[i].L(p, n, w);
}

unsigned int TraceResult::LightIndex() const
{
	unsigned int i = g_SceneData.m_sMatData[m_pTri->getMatIndex(m_pNode->m_uMaterialOffset)].NodeLightIndex;
	if(i == 0xffffffff)
		return 0xffffffff;
	unsigned int j = m_pNode->m_uLightIndices[i];
	return j;
}

const e_KernelMaterial& TraceResult::getMat() const
{
	return g_SceneData.m_sMatData[getMatIndex()];
}