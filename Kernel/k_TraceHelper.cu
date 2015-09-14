#include "k_TraceHelper.h"
#include "../Math/Compression.h"
#include "../Math/half.h"
#include "cuda_runtime.h"
#include "../Engine/e_Sensor.h"
#include "../Engine/e_Mesh.h"
#include "../Engine/e_TriangleData.h"
#include "../Engine/e_Material.h"
#include "../Engine/e_IntersectorData.h"
#include "../Engine/e_Node.h"
#include "../Engine/e_DynamicScene.h"

//#define SKIP_OUTER_TREE

enum
{
	MaxBlockHeight = 6,            // Upper bound for blockDim.y.
	EntrypointSentinel = 0x76543210,   // Bottom-most stack entry, indicating the end of traversal.
};

e_KernelDynamicScene g_SceneDataDevice;
unsigned int g_RayTracedCounterDevice;
e_Sensor g_CameraDataDevice;
CudaRNGBuffer g_RNGDataDevice;

e_KernelDynamicScene g_SceneDataHost;
unsigned int g_RayTracedCounterHost;
e_Sensor g_CameraDataHost;
CudaRNGBuffer g_RNGDataHost;

texture<float4, 1>		t_nodesA;
texture<float4, 1>		t_tris;
texture<unsigned int,  1>		t_triIndices;
texture<float4, 1>		t_SceneNodes;
texture<float4, 1>		t_NodeTransforms;
texture<float4, 1>		t_NodeInvTransforms;

texture<int2, 1> t_TriDataA;
texture<float4, 1> t_TriDataB;

void traversalResult::toResult(TraceResult* tR, e_KernelDynamicScene& data)
{
	tR->m_fDist = dist;
	tR->m_fBaryCoords = ((half2*)&bCoords)->ToFloat2();
	tR->m_pNode = data.m_sNodeData.Data + nodeIdx;
	tR->m_pTri = data.m_sTriData.Data + triIdx;
}

CUDA_FUNC_IN void loadModl(int i, float4x4* o)
{
#ifdef ISCUDA
	float4* f = (float4*)o;
	f[0] = tex1Dfetch(t_NodeTransforms, i * 4 + 0);
	f[1] = tex1Dfetch(t_NodeTransforms, i * 4 + 1);
	f[2] = tex1Dfetch(t_NodeTransforms, i * 4 + 2);
	f[3] = tex1Dfetch(t_NodeTransforms, i * 4 + 3);
#else
	*o = g_SceneData.m_sSceneBVH.m_pNodeTransforms[i];
#endif
}

CUDA_FUNC_IN void loadInvModl(int i, float4x4* o)
{
#ifdef ISCUDA
	float4* f = (float4*)o;
	f[0] = tex1Dfetch(t_NodeInvTransforms, i * 4 + 0);
	f[1] = tex1Dfetch(t_NodeInvTransforms, i * 4 + 1);
	f[2] = tex1Dfetch(t_NodeInvTransforms, i * 4 + 2);
	f[3] = tex1Dfetch(t_NodeInvTransforms, i * 4 + 3);
#else
	*o = g_SceneData.m_sSceneBVH.m_pInvNodeTransforms[i];
#endif
}

CUDA_FUNC_IN bool k_TraceRayNode(const Vec3f& dir, const Vec3f& ori, TraceResult* a_Result, const e_Node* N)
{
	unsigned int mIndex = N->m_uMeshIndex;
	e_KernelMesh mesh = g_SceneData.m_sMeshData[mIndex];
	bool found = false;
	int traversalStack[64];
	traversalStack[0] = EntrypointSentinel;
	float   dirx = dir.x;
	float   diry = dir.y;
	float   dirz = dir.z;
	const float ooeps = math::exp2(-80.0f);
	float   idirx = 1.0f / (math::abs(dir.x) > ooeps ? dir.x : copysignf(ooeps, dir.x));
	float   idiry = 1.0f / (math::abs(dir.y) > ooeps ? dir.y : copysignf(ooeps, dir.y));
	float   idirz = 1.0f / (math::abs(dir.z) > ooeps ? dir.z : copysignf(ooeps, dir.z));
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
			Vec4f* dat = (Vec4f*)g_SceneData.m_sBVHNodeData.Data;
			const Vec4f n0xy = dat[mesh.m_uBVHNodeOffset + nodeAddr + 0];
			const Vec4f n1xy = dat[mesh.m_uBVHNodeOffset + nodeAddr + 1];
			const Vec4f nz   = dat[mesh.m_uBVHNodeOffset + nodeAddr + 2];
				  Vec4f tmp  = dat[mesh.m_uBVHNodeOffset + nodeAddr + 3];
#endif
				  Vec2i  cnodes = *(Vec2i*)&tmp;
			const float c0lox = n0xy.x * idirx - oodx;
			const float c0hix = n0xy.y * idirx - oodx;
			const float c0loy = n0xy.z * idiry - oody;
			const float c0hiy = n0xy.w * idiry - oody;
			const float c0loz = nz.x   * idirz - oodz;
			const float c0hiz = nz.y   * idirz - oodz;
			const float c1loz = nz.z   * idirz - oodz;
			const float c1hiz = nz.w   * idirz - oodz;
			const float c0min = math::spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, 0);
			const float c0max = math::spanEndKepler  (c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, a_Result->m_fDist);
			const float c1lox = n1xy.x * idirx - oodx;
			const float c1hix = n1xy.y * idirx - oodx;
			const float c1loy = n1xy.z * idiry - oody;
			const float c1hiy = n1xy.w * idiry - oody;
			const float c1min = math::spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, 0);
			const float c1max = math::spanEndKepler  (c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, a_Result->m_fDist);
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

#ifdef ISCUDA
            unsigned int mask;
            asm("{\n"
                "   .reg .pred p;               \n"
                "setp.ge.s32        p, %1, 0;   \n"
                "vote.ballot.b32    %0,p;       \n"
                "}"
                : "=r"(mask)
                : "r"(leafAddr));
#else
			unsigned int mask = leafAddr >= 0;
#endif
			if(!mask)
				break;
		}
		while (leafAddr < 0)
		{
			if (leafAddr != -214783648)
			{
				for (int triAddr = ~leafAddr;; triAddr++)
				{
#ifdef ISCUDA
					const float4 v00 = tex1Dfetch(t_tris, mesh.m_uBVHTriangleOffset + triAddr * 3 + 0);
					const float4 v11 = tex1Dfetch(t_tris, mesh.m_uBVHTriangleOffset + triAddr * 3 + 1);
					const float4 v22 = tex1Dfetch(t_tris, mesh.m_uBVHTriangleOffset + triAddr * 3 + 2);
					unsigned int index = tex1Dfetch(t_triIndices, mesh.m_uBVHIndicesOffset + triAddr);
#else
					Vec4f* dat = (Vec4f*)g_SceneData.m_sBVHIntData.Data;
					const Vec4f v00 = dat[mesh.m_uBVHTriangleOffset + triAddr * 3 + 0];
					const Vec4f v11 = dat[mesh.m_uBVHTriangleOffset + triAddr * 3 + 1];
					const Vec4f v22 = dat[mesh.m_uBVHTriangleOffset + triAddr * 3 + 2];
					unsigned int index = g_SceneData.m_sBVHIndexData.Data[mesh.m_uBVHIndicesOffset + triAddr].index;
#endif

					float Oz = v00.w - origx*v00.x - origy*v00.y - origz*v00.z;
					float invDz = 1.0f / (dirx*v00.x + diry*v00.y + dirz*v00.z);
					float t = Oz * invDz;
					if (t > 1e-2f && t < a_Result->m_fDist)
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
								unsigned int ti = index >> 1;
								e_TriangleData* tri = g_SceneData.m_sTriData.Data + ti + mesh.m_uTriangleOffset;
								int q = 1;
								/*if (USE_ALPHA)
								{
									e_KernelMaterial* mat = g_SceneData.m_sMatData.Data + tri->getMatIndex(N->m_uMaterialOffset);
									DifferentialGeometry dg;
									dg.bary = make_float2(u, v);
									for (int i = 0; i < NUM_UV_SETS; i++)
										dg.uv[i] = tri->math::lerpUV(i, dg.bary);
									float a = mat->SampleAlphaMap(dg);
									q = a >= mat->m_fAlphaThreshold;

								}*/
								if (q)
								{
									a_Result->m_pNode = N;
									a_Result->m_pTri = tri;
									a_Result->m_fBaryCoords = Vec2f(u, v);
									a_Result->m_fDist = t;
									found = true;
								}
							}
						}
					}
					if (index & 1)
						break;
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

bool k_TraceRay(const Vec3f& dir, const Vec3f& ori, TraceResult* a_Result)
{
	Platform::Increment(&g_RayTracedCounter);
	if(!g_SceneData.m_sNodeData.UsedCount)
		return false;
#ifdef SKIP_OUTER_TREE
	const int node = 0;
	e_Node* N = g_SceneData.m_sNodeData.Data + node;
	//transform a_Result->m_fDist to local system
	float4x4 modl;
	loadInvModl(node, &modl);
	Vec3f d = modl.TransformDirection(dir), o = modl.TransformPoint(ori);
	k_TraceRayNode(d, o, a_Result, N);
#else
	int traversalStackOuter[64];
	int at = 1;
	traversalStackOuter[0] = g_SceneData.m_sSceneBVH.m_sStartNode;
	const float ooeps = math::exp2(-80.0f);
	Vec3f O, I;
	I.x = 1.0f / (math::abs(dir.x) > ooeps ? dir.x : copysignf(ooeps, dir.x));
	I.y = 1.0f / (math::abs(dir.y) > ooeps ? dir.y : copysignf(ooeps, dir.y));
	I.z = 1.0f / (math::abs(dir.z) > ooeps ? dir.z : copysignf(ooeps, dir.z));
	O = I * ori;
	while(at)
	{
		int nodeAddrOuter = traversalStackOuter[--at];
		while (nodeAddrOuter >= 0 && nodeAddrOuter != EntrypointSentinel)
		{
#ifdef ISCUDA
			const float4 n0xy = tex1Dfetch(t_SceneNodes, nodeAddrOuter + 0); // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
			const float4 n1xy = tex1Dfetch(t_SceneNodes, nodeAddrOuter + 1); // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
			const float4 nz   = tex1Dfetch(t_SceneNodes, nodeAddrOuter + 2); // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
				  float4 tmp  = tex1Dfetch(t_SceneNodes, nodeAddrOuter + 3); // child_index0, child_index1
#else
			const Vec4f n0xy = g_SceneData.m_sSceneBVH.m_pNodes[nodeAddrOuter / 4].a;
			const Vec4f n1xy = g_SceneData.m_sSceneBVH.m_pNodes[nodeAddrOuter / 4].b;
			const Vec4f nz   = g_SceneData.m_sSceneBVH.m_pNodes[nodeAddrOuter / 4].c;
				  Vec4f tmp  = g_SceneData.m_sSceneBVH.m_pNodes[nodeAddrOuter / 4].d;
#endif
			Vec2i  cnodesOuter = *(Vec2i*)&tmp;
			const float c0lox = n0xy.x * I.x - O.x;
			const float c0hix = n0xy.y * I.x - O.x;
			const float c0loy = n0xy.z * I.y - O.y;
			const float c0hiy = n0xy.w * I.y - O.y;
			const float c0loz = nz.x   * I.z - O.z;
			const float c0hiz = nz.y   * I.z - O.z;
			const float c1loz = nz.z   * I.z - O.z;
			const float c1hiz = nz.w   * I.z - O.z;
			const float c0min = math::spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, 0);
			const float c0max = math::spanEndKepler  (c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, a_Result->m_fDist);
			const float c1lox = n1xy.x * I.x - O.x;
			const float c1hix = n1xy.y * I.x - O.x;
			const float c1loy = n1xy.z * I.y - O.y;
			const float c1hiy = n1xy.w * I.y - O.y;
			const float c1min = math::spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, 0);
			const float c1max = math::spanEndKepler  (c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, a_Result->m_fDist);
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
		if(nodeAddrOuter < 0 && nodeAddrOuter != -214783648)
		{
			int node = ~nodeAddrOuter;
			e_Node* N = g_SceneData.m_sNodeData.Data + node;
			//transform a_Result->m_fDist to local system
			float4x4 modl, modl2;
			loadInvModl(node, &modl);
			loadModl(node, &modl2);
			Vec3f d = modl.TransformDirection(dir), o = modl.TransformPoint(ori);
			k_TraceRayNode(d, o, a_Result, N);
		}
	}
#endif
	return a_Result->hasHit();
}

void k_INITIALIZE(e_DynamicScene* a_Scene, const CudaRNGBuffer& a_RngBuf)
{
	if (!a_Scene)
		return;

	e_KernelDynamicScene a_Data = a_Scene->getKernelSceneData();

	size_t offset;
	cudaChannelFormatDesc	cdf4 = cudaCreateChannelDesc<float4>(),
							cdu1 = cudaCreateChannelDesc<unsigned int>(),
							cdi2 = cudaCreateChannelDesc<int2>(),
							cdh4 = cudaCreateChannelDescHalf4();
	cudaError_t
	r = cudaBindTexture(&offset, &t_nodesA, a_Data.m_sBVHNodeData.Data, &cdf4, a_Data.m_sBVHNodeData.UsedCount * sizeof(e_BVHNodeData));
	r = cudaBindTexture(&offset, &t_tris, a_Data.m_sBVHIntData.Data, &cdf4, a_Data.m_sBVHIntData.UsedCount * sizeof(e_TriIntersectorData));
	r = cudaBindTexture(&offset, &t_triIndices, a_Data.m_sBVHIndexData.Data, &cdu1, a_Data.m_sBVHIndexData.UsedCount * sizeof(e_TriIntersectorData2));
	r = cudaBindTexture(&offset, &t_SceneNodes, a_Data.m_sSceneBVH.m_pNodes, &cdf4, a_Data.m_sSceneBVH.m_uNumNodes * sizeof(e_BVHNodeData));
	r = cudaBindTexture(&offset, &t_NodeTransforms, a_Data.m_sSceneBVH.m_pNodeTransforms, &cdf4, a_Data.m_sNodeData.UsedCount * sizeof(float4x4));
	r = cudaBindTexture(&offset, &t_NodeInvTransforms, a_Data.m_sSceneBVH.m_pInvNodeTransforms, &cdf4, a_Data.m_sNodeData.UsedCount * sizeof(float4x4));

	r = cudaBindTexture(&offset, &t_TriDataA, a_Data.m_sTriData.Data, &cdi2, a_Data.m_sTriData.UsedCount * sizeof(e_TriangleData));
	r = cudaBindTexture(&offset, &t_TriDataB, a_Data.m_sTriData.Data, &cdh4, a_Data.m_sTriData.UsedCount * sizeof(e_TriangleData));

	unsigned int b = 0;
	cudaMemcpyToSymbol(g_RayTracedCounterDevice, &b, sizeof(unsigned int));
	cudaMemcpyToSymbol(g_SceneDataDevice, &a_Data, sizeof(e_KernelDynamicScene));
	cudaMemcpyToSymbol(g_RNGDataDevice, &a_RngBuf, sizeof(CudaRNGBuffer));

	g_SceneDataHost = a_Scene->getKernelSceneData(false);
	g_RNGDataHost = a_RngBuf;
	g_RayTracedCounterHost = 0;
}

void fillDG(const Vec2f& bary, const e_TriangleData* tri, const e_Node* node, DifferentialGeometry& dg)
{
	float4x4 localToWorld, worldToLocal;
	loadModl(node - g_SceneData.m_sNodeData.Data, &localToWorld);
	loadInvModl(node - g_SceneData.m_sNodeData.Data, &worldToLocal);
	dg.bary = bary;
	dg.hasUVPartials = false;
#if defined(ISCUDA) && NUM_UV_SETS == 1
	unsigned int i = tri - g_SceneData.m_sTriData.Data;
	int2 nme = tex1Dfetch(t_TriDataA, i * 4 + 0);
	float4 rowB = tex1Dfetch(t_TriDataB, i * 4 + 1);
	float4 rowC = tex1Dfetch(t_TriDataB, i * 4 + 2);
	float4 rowD = tex1Dfetch(t_TriDataB, i * 4 + 3);
	Vec3f na = Uchar2ToNormalizedFloat3(nme.x), nb = Uchar2ToNormalizedFloat3(nme.x >> 16), nc = Uchar2ToNormalizedFloat3(nme.y);
	float w = 1.0f - dg.bary.x - dg.bary.y, u = dg.bary.x, v = dg.bary.y;
	dg.extraData = nme.y >> 24;
	dg.sys.n = u * na + v * nb + w * nc;
	Vec3f dpdu = Vec3f(rowB.x, rowB.y, rowB.z);
	Vec3f dpdv = Vec3f(rowB.z, rowC.x, rowC.y);
	dg.sys.s = dpdu - dg.sys.n * dot(dg.sys.n, dpdu);
	dg.sys.t = cross(dg.sys.s, dg.sys.n);
	dg.sys = dg.sys * localToWorld;
	dg.n = normalize(worldToLocal.TransformTranspose(Vec4f(na + nb + nc, 0.0f)).getXYZ());
	dg.dpdu = localToWorld.TransformDirection(dpdu);
	dg.dpdv = localToWorld.TransformDirection(dpdv);
	Vec2f ta = Vec2f(rowC.z, rowC.w), tb = Vec2f(rowD.x, rowD.y), tc = Vec2f(rowD.z, rowD.w);
	dg.uv[0] = u * ta + v * tb + w * tc;

	if (dot(dg.n, dg.sys.n) < 0.0f)
		dg.n = -dg.n;
#else
	tri->fillDG(localToWorld, worldToLocal, dg);
#endif
}

unsigned int k_getNumRaysTraced()
{
	unsigned int i;
	cudaMemcpyFromSymbol(&i, g_RayTracedCounterDevice, sizeof(unsigned int));
	return i + g_RayTracedCounterHost;
}

void k_setNumRaysTraced(unsigned int i)
{
	g_RayTracedCounterHost = i;
	cudaMemcpyToSymbol(g_RayTracedCounterDevice, &i, sizeof(unsigned int));
}

#define DYNAMIC_FETCH_THRESHOLD 20
#define STACK_SIZE 32
__device__ int g_warpCounter;
__device__ __inline__ int   min_min2   (int a, int b, int c) { int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   min_max2   (int a, int b, int c) { int v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_min2   (int a, int b, int c) { int v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_max2   (int a, int b, int c) { int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ float fmin_fmin2 (float a, float b, float c) { return __int_as_float(min_min2(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmin_fmax2 (float a, float b, float c) { return __int_as_float(min_max2(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmin2 (float a, float b, float c) { return __int_as_float(max_min2(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmax2 (float a, float b, float c) { return __int_as_float(max_max2(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float spanBeginKepler2(float a0, float a1, float b0, float b1, float c0, float c1, float d){	return fmax_fmax2( min(a0,a1), min(b0,b1), fmin_fmax2(c0, c1, d)); }
__device__ __inline__ float spanEndKepler2(float a0, float a1, float b0, float b1, float c0, float c1, float d)	{	return fmin_fmin2( max(a0,a1), max(b0,b1), fmax_fmin2(c0, c1, d)); }

template<bool ANY_HIT> __global__ void intersectKernel_SKIPOUTER(int numRays, traversalRay* a_RayBuffer, traversalResult* a_ResBuffer)
{
    int traversalStack[STACK_SIZE];
    traversalStack[0] = EntrypointSentinel; // Bottom-most entry.

    // Live state during traversal, stored in registers.

    float   origx, origy, origz;            // Ray origin.
    char*   stackPtr;                       // Current position in traversal stack.
    int     leafAddr;                       // First postponed leaf, non-negative if none.
    int     nodeAddr = EntrypointSentinel;  // Non-negative: current internal node, negative: second postponed leaf.
    int     hitIndex;                       // Triangle index of the closest intersection, -1 if none.
    float   hitT;                           // t-value of the closest intersection.
    float   tmin;
    int     rayidx;
    float   oodx;
    float   oody;
    float   oodz;
    float   dirx;
    float   diry;
    float   dirz;
    float   idirx;
    float   idiry;
    float   idirz;
	Vec2f bCoords;

    // Initialize persistent threads.

    __shared__ volatile int nextRayArray[MaxBlockHeight]; // Current ray index in global buffer.

    // Persistent threads: fetch and process rays in a loop.

    do
    {
        const int tidx = threadIdx.x;
        volatile int& rayBase = nextRayArray[threadIdx.y];

        // Fetch new rays from the global pool using lane 0.

        const bool          terminated     = nodeAddr==EntrypointSentinel;
        const unsigned int  maskTerminated = __ballot(terminated);
        const int           numTerminated  = __popc(maskTerminated);
        const int           idxTerminated  = __popc(maskTerminated & ((1u<<tidx)-1));

        if(terminated)
        {
            if (idxTerminated == 0)
                rayBase = atomicAdd(&g_warpCounter, numTerminated);

            rayidx = rayBase + idxTerminated;
            if (rayidx >= numRays)
                break;

            // Fetch ray.

			float4 o1 = ((float4*)a_RayBuffer)[rayidx * 2 + 0];
			float4 d1 = ((float4*)a_RayBuffer)[rayidx * 2 + 1];
			//to local
			float4x4 modl;
			loadInvModl(0, &modl);
			float3 d = modl.TransformDirection(Vec3f(d1.x, d1.y, d1.z)), o = modl.TransformPoint(Vec3f(o1.x, o1.y, o1.z));

            origx = o.x;
            origy = o.y;
            origz = o.z;
			//tmin  = o1.w / length(d);
			tmin = o1.w;
            dirx  = d.x;
            diry  = d.y;
            dirz  = d.z;
            //hitT  = d1.w / length(d);
			hitT = d1.w;
			float ooeps = math::exp2(-80.0f); // Avoid div by zero.
            idirx = 1.0f / (math::abs(d.x) > ooeps ? d.x : copysignf(ooeps, d.x));
            idiry = 1.0f / (math::abs(d.y) > ooeps ? d.y : copysignf(ooeps, d.y));
            idirz = 1.0f / (math::abs(d.z) > ooeps ? d.z : copysignf(ooeps, d.z));
            oodx  = origx * idirx;
            oody  = origy * idiry;
            oodz  = origz * idirz;

            // Setup traversal.

            stackPtr = (char*)&traversalStack[0];
            leafAddr = 0;   // No postponed leaf.
            nodeAddr = 0;   // Start from the root.
            hitIndex = -1;  // No triangle intersected so far.
		}
		
		while(nodeAddr != EntrypointSentinel)
		{
			while (unsigned int(nodeAddr) < unsigned int(EntrypointSentinel))
			{
				const float4 n0xy = tex1Dfetch(t_nodesA, nodeAddr + 0); // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
				const float4 n1xy = tex1Dfetch(t_nodesA, nodeAddr + 1); // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
				const float4 nz   = tex1Dfetch(t_nodesA, nodeAddr + 2); // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
					  float4 tmp  = tex1Dfetch(t_nodesA, nodeAddr + 3); // child_index0, child_index1
						int2  cnodes= *(int2*)&tmp;

				// Intersect the ray against the child nodes.

                const float c0lox = n0xy.x * idirx - oodx;
                const float c0hix = n0xy.y * idirx - oodx;
                const float c0loy = n0xy.z * idiry - oody;
                const float c0hiy = n0xy.w * idiry - oody;
                const float c0loz = nz.x   * idirz - oodz;
                const float c0hiz = nz.y   * idirz - oodz;
                const float c1loz = nz.z   * idirz - oodz;
                const float c1hiz = nz.w   * idirz - oodz;
                const float c0min = spanBeginKepler2(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, tmin);
                const float c0max = spanEndKepler2  (c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hitT);
                const float c1lox = n1xy.x * idirx - oodx;
                const float c1hix = n1xy.y * idirx - oodx;
                const float c1loy = n1xy.z * idiry - oody;
                const float c1hiy = n1xy.w * idiry - oody;
                const float c1min = spanBeginKepler2(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, tmin);
                const float c1max = spanEndKepler2  (c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, hitT);

                bool swp = (c1min < c0min);

                bool traverseChild0 = (c0max >= c0min);
                bool traverseChild1 = (c1max >= c1min);

                // Neither child was intersected => pop stack.

                if (!traverseChild0 && !traverseChild1)
                {
                    nodeAddr = *(int*)stackPtr;
                    stackPtr -= 4;
                }

                // Otherwise => fetch child pointers.

                else
                {
                    nodeAddr = (traverseChild0) ? cnodes.x : cnodes.y;

                    // Both children were intersected => push the farther one.

                    if (traverseChild0 && traverseChild1)
                    {
                        if (swp)
                            swapk(nodeAddr, cnodes.y);
                        stackPtr += 4;
                        *(int*)stackPtr = cnodes.y;
                    }
                }

                // First leaf => postpone and continue traversal.

                if (nodeAddr < 0 && leafAddr  >= 0)     // Postpone max 1
//              if (nodeAddr < 0 && leafAddr2 >= 0)     // Postpone max 2
                {
                    //leafAddr2= leafAddr;          // postpone 2
                    leafAddr = nodeAddr;
                    nodeAddr = *(int*)stackPtr;
                    stackPtr -= 4;
                }

                // All SIMD lanes have found a leaf? => process them.

                // NOTE: inline PTX implementation of "if(!__any(leafAddr >= 0)) break;".
                // tried everything with CUDA 4.2 but always got several redundant instructions.

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
				for (int triAddr = ~leafAddr;; triAddr++)
				{
					// Tris in TEX (good to fetch as a single batch)
					const float4 v00 = tex1Dfetch(t_tris, triAddr * 3 + 0);
					const float4 v11 = tex1Dfetch(t_tris, triAddr * 3 + 1);
					const float4 v22 = tex1Dfetch(t_tris, triAddr * 3 + 2);
					unsigned int index = tex1Dfetch(t_triIndices, triAddr);

					float Oz = v00.w - origx*v00.x - origy*v00.y - origz*v00.z;
					float invDz = 1.0f / (dirx*v00.x + diry*v00.y + dirz*v00.z);
					float t = Oz * invDz;

					if (t > tmin && t < hitT)
					{
						// Compute and check barycentric u.

						float Ox = v11.w + origx*v11.x + origy*v11.y + origz*v11.z;
						float Dx = dirx*v11.x + diry*v11.y + dirz*v11.z;
						float u = Ox + t*Dx;

						if (u >= 0.0f)
						{
							// Compute and check barycentric v.

							float Oy = v22.w + origx*v22.x + origy*v22.y + origz*v22.z;
							float Dy = dirx*v22.x + diry*v22.y + dirz*v22.z;
							float v = Oy + t*Dy;

							if (v >= 0.0f && u + v <= 1.0f)
							{
								// Record intersection.
								// Closest intersection not required => terminate.

								hitT = t;
								hitIndex = index >> 1;
								bCoords = Vec2f(u,v);
								if (ANY_HIT)
								{
									nodeAddr = EntrypointSentinel;
									break;
								}
							}
						}
					}
					if(index & 1)
						break;
				} // triangle

				leafAddr = nodeAddr;
				if (nodeAddr < 0)
				{
					nodeAddr = *(int*)stackPtr;
					stackPtr -= 4;
				}
			}
			if( __popc(__ballot(true)) < DYNAMIC_FETCH_THRESHOLD )
				break;
		}
		Vec4i res = Vec4i(0, 0, 0, 0);
		if(hitIndex != -1)
		{
			//res.x = __float_as_int(hitT * math::sqrt(dirx * dirx + diry * diry + dirz * dirz));
			res.x = __float_as_int(hitT);
			res.y = 0;
			res.z = hitIndex;
			half2 h(bCoords);
			res.w = *(int*)&h;
		}
		((int4*)a_ResBuffer)[rayidx] = res;
	} while(true);
}

template<bool ANY_HIT> __global__ void intersectKernel(int numRays, traversalRay* a_RayBuffer, traversalResult* a_ResBuffer)
{
	// Traversal stack in CUDA thread-local memory.

    int traversalStack[STACK_SIZE];
    traversalStack[0] = EntrypointSentinel; // Bottom-most entry.

    // Live state during traversal, stored in registers.

    float   origx, origy, origz;            // Ray origin.
    char*   stackPtr;                       // Current position in traversal stack.
    int     leafAddr;                       // First postponed leaf, non-negative if none.
    int     nodeAddr = EntrypointSentinel;  // Non-negative: current internal node, negative: second postponed leaf.
    int     hitIndex;                       // Triangle index of the closest intersection, -1 if none.
    float   hitT;                           // t-value of the closest intersection.
    float   tmin;
    int     rayidx;
    float   oodx;
    float   oody;
    float   oodz;
    float   dirx;
    float   diry;
    float   dirz;
    float   idirx;
    float   idiry;
    float   idirz;
	Vec2f bCorrds;
	int nodeIdx = 0;

	int ltraversalStack[STACK_SIZE];
	ltraversalStack[0] = EntrypointSentinel;
    float   lorigx, lorigy, lorigz;
    char*   lstackPtr;
    int     lleafAddr;
    int     lnodeAddr = EntrypointSentinel;
    float   lhitT;
    float   ltmin;
    float   loodx;
    float   loody;
    float   loodz;
    float   ldirx;
    float   ldiry;
    float   ldirz;
    float   lidirx;
    float   lidiry;
    float   lidirz;

    // Initialize persistent threads.

    __shared__ volatile int nextRayArray[MaxBlockHeight]; // Current ray index in global buffer.

    // Persistent threads: fetch and process rays in a loop.

    do
    {
        const int tidx = threadIdx.x;
        volatile int& rayBase = nextRayArray[threadIdx.y];

        // Fetch new rays from the global pool using lane 0.

        const bool          terminated     = nodeAddr==EntrypointSentinel;
        const unsigned int  maskTerminated = __ballot(terminated);
        const int           numTerminated  = __popc(maskTerminated);
        const int           idxTerminated  = __popc(maskTerminated & ((1u<<tidx)-1));

        if(terminated)
        {
            if (idxTerminated == 0)
                rayBase = atomicAdd(&g_warpCounter, numTerminated);

            rayidx = rayBase + idxTerminated;
            if (rayidx >= numRays)
                break;

            // Fetch ray.

			float4 o = ((float4*)a_RayBuffer)[rayidx * 2 + 0];
			float4 d = ((float4*)a_RayBuffer)[rayidx * 2 + 1];
            origx = o.x;
            origy = o.y;
            origz = o.z;
            tmin  = o.w;
            dirx  = d.x;
            diry  = d.y;
            dirz  = d.z;
            hitT  = d.w;
			float ooeps = math::exp2(-80.0f); // Avoid div by zero.
            idirx = 1.0f / (math::abs(d.x) > ooeps ? d.x : copysignf(ooeps, d.x));
            idiry = 1.0f / (math::abs(d.y) > ooeps ? d.y : copysignf(ooeps, d.y));
            idirz = 1.0f / (math::abs(d.z) > ooeps ? d.z : copysignf(ooeps, d.z));
            oodx  = origx * idirx;
            oody  = origy * idiry;
            oodz  = origz * idirz;

            // Setup traversal.

            stackPtr = (char*)&traversalStack[0];
            leafAddr = 0;   // No postponed leaf.
            leafAddr = nodeAddr = g_SceneData.m_sSceneBVH.m_sStartNode;   // Start from the root. set the leafAddr to support scenes with one node
            hitIndex = -1;  // No triangle intersected so far.
        }

        // Traversal loop.
		TraceResult r2 = k_TraceRay(Ray(a_RayBuffer[rayidx].a.getXYZ(), a_RayBuffer[rayidx].b.getXYZ()));
		int4 res = make_int4(0, 0, 0, 0);
		if (r2.hasHit())
		{
			res.x = __float_as_int(r2.m_fDist);
			res.y = r2.getNodeIndex();
			res.z = r2.m_pTri - g_SceneData.m_sTriData.Data;
			half2 h(r2.m_fBaryCoords);
			res.w = *(int*)&h;
		}
		((int4*)a_ResBuffer)[rayidx] = res;
		nodeAddr = EntrypointSentinel;

		/*if (g_SceneData.m_sNodeData.UsedCount == 0)
			nodeAddr = EntrypointSentinel;

		while (nodeAddr != EntrypointSentinel)
        {
			//nodeAddr = nodeAddr == EntrypointSentinel - 1 ? EntrypointSentinel : nodeAddr;
            // Traverse internal nodes until all SIMD lanes have found a leaf.

            while (unsigned int(nodeAddr) < unsigned int(EntrypointSentinel))   // functionally equivalent, but faster
            {
                // Fetch AABBs of the two child nodes.

                const float4 n0xy = tex1Dfetch(t_SceneNodes, nodeAddr + 0); // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
                const float4 n1xy = tex1Dfetch(t_SceneNodes, nodeAddr + 1); // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
                const float4 nz   = tex1Dfetch(t_SceneNodes, nodeAddr + 2); // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
                      float4 tmp  = tex1Dfetch(t_SceneNodes, nodeAddr + 3); // child_index0, child_index1
                      int2  cnodes= *(int2*)&tmp;

                // Intersect the ray against the child nodes.

                const float c0lox = n0xy.x * idirx - oodx;
                const float c0hix = n0xy.y * idirx - oodx;
                const float c0loy = n0xy.z * idiry - oody;
                const float c0hiy = n0xy.w * idiry - oody;
                const float c0loz = nz.x   * idirz - oodz;
                const float c0hiz = nz.y   * idirz - oodz;
                const float c1loz = nz.z   * idirz - oodz;
                const float c1hiz = nz.w   * idirz - oodz;
                const float c0min = spanBeginKepler2(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, tmin);
                const float c0max = spanEndKepler2  (c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hitT);
                const float c1lox = n1xy.x * idirx - oodx;
                const float c1hix = n1xy.y * idirx - oodx;
                const float c1loy = n1xy.z * idiry - oody;
                const float c1hiy = n1xy.w * idiry - oody;
                const float c1min = spanBeginKepler2(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, tmin);
                const float c1max = spanEndKepler2  (c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, hitT);

                bool swp = (c1min < c0min);

                bool traverseChild0 = (c0max >= c0min);
                bool traverseChild1 = (c1max >= c1min);

                // Neither child was intersected => pop stack.

                if (!traverseChild0 && !traverseChild1)
                {
                    nodeAddr = *(int*)stackPtr;
                    stackPtr -= 4;
                }
                else// Otherwise => fetch child pointers.
                {
                    nodeAddr = (traverseChild0) ? cnodes.x : cnodes.y;

                    // Both children were intersected => push the farther one.

                    if (traverseChild0 && traverseChild1)
                    {
                        if (swp)
                            swapk(nodeAddr, cnodes.y);
                        stackPtr += 4;
                        *(int*)stackPtr = cnodes.y;
                    }
                }

                // First leaf => postpone and continue traversal.

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

            // Process postponed leaf nodes.

            while (leafAddr < 0)
            {
				e_Node* N = g_SceneData.m_sNodeData.Data + (~leafAddr);
				if (terminated)
				{
					float4x4 modl;
					loadInvModl(~leafAddr, &modl);
					float3 d = modl.TransformDirection(Vec3f(dirx, diry, dirz)), o = modl.TransformPoint(Vec3f(origx, origy, origz));

					lorigx = o.x;
					lorigy = o.y;
					lorigz = o.z;
					ltmin = tmin;
					ldirx = d.x;
					ldiry = d.y;
					ldirz = d.z;
					lhitT = hitT;
					float ooeps = math::exp2(-80.0f); // Avoid div by zero.
					lidirx = 1.0f / (math::abs(d.x) > ooeps ? d.x : copysignf(ooeps, d.x));
					lidiry = 1.0f / (math::abs(d.y) > ooeps ? d.y : copysignf(ooeps, d.y));
					lidirz = 1.0f / (math::abs(d.z) > ooeps ? d.z : copysignf(ooeps, d.z));
					loodx = lorigx * lidirx;
					loody = lorigy * lidiry;
					loodz = lorigz * lidirz;
					lstackPtr = (char*)&ltraversalStack[0];
					lleafAddr = 0;   // No postponed leaf.
					lnodeAddr = 0;   // Start from the root.
				}

				unsigned int m_uBVHNodeOffset = g_SceneData.m_sMeshData[N->m_uMeshIndex].m_uBVHNodeOffset,
					m_uBVHTriangleOffset = g_SceneData.m_sMeshData[N->m_uMeshIndex].m_uBVHTriangleOffset,
					m_uBVHIndicesOffset = g_SceneData.m_sMeshData[N->m_uMeshIndex].m_uBVHIndicesOffset,
					m_uTriangleOffset = g_SceneData.m_sMeshData[N->m_uMeshIndex].m_uTriangleOffset;

				while (lnodeAddr != EntrypointSentinel)
				{
					while (unsigned int(lnodeAddr) < unsigned int(EntrypointSentinel))
					{
						const float4 n0xy = tex1Dfetch(t_nodesA, lnodeAddr + 0 + m_uBVHNodeOffset); // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
						const float4 n1xy = tex1Dfetch(t_nodesA, lnodeAddr + 1 + m_uBVHNodeOffset); // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
						const float4 nz = tex1Dfetch(t_nodesA, lnodeAddr + 2 + m_uBVHNodeOffset); // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
						float4 tmp = tex1Dfetch(t_nodesA, lnodeAddr + 3 + m_uBVHNodeOffset); // child_index0, child_index1
						int2  cnodes = *(int2*)&tmp;

						// Intersect the ray against the child nodes.

						const float c0lox = n0xy.x * lidirx - loodx;
						const float c0hix = n0xy.y * lidirx - loodx;
						const float c0loy = n0xy.z * lidiry - loody;
						const float c0hiy = n0xy.w * lidiry - loody;
						const float c0loz = nz.x   * lidirz - loodz;
						const float c0hiz = nz.y   * lidirz - loodz;
						const float c1loz = nz.z   * lidirz - loodz;
						const float c1hiz = nz.w   * lidirz - loodz;
						const float c0min = spanBeginKepler2(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, ltmin);
						const float c0max = spanEndKepler2(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, lhitT);
						const float c1lox = n1xy.x * lidirx - loodx;
						const float c1hix = n1xy.y * lidirx - loodx;
						const float c1loy = n1xy.z * lidiry - loody;
						const float c1hiy = n1xy.w * lidiry - loody;
						const float c1min = spanBeginKepler2(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, ltmin);
						const float c1max = spanEndKepler2(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, lhitT);

						bool swp = (c1min < c0min);

						bool traverseChild0 = (c0max >= c0min);
						bool traverseChild1 = (c1max >= c1min);

						// Neither child was intersected => pop stack.

						if (!traverseChild0 && !traverseChild1)
						{
							lnodeAddr = *(int*)lstackPtr;
							lstackPtr -= 4;
						}
						else// Otherwise => fetch child pointers.
						{
							lnodeAddr = (traverseChild0) ? cnodes.x : cnodes.y;

							// Both children were intersected => push the farther one.

							if (traverseChild0 && traverseChild1)
							{
								if (swp)
									swapk(lnodeAddr, cnodes.y);
								lstackPtr += 4;
								*(int*)lstackPtr = cnodes.y;
							}
						}

						if (lnodeAddr < 0 && lleafAddr >= 0)     // Postpone max 1
						{
							lleafAddr = lnodeAddr;
							lnodeAddr = *(int*)lstackPtr;
							lstackPtr -= 4;
						}

						unsigned int mask;
						asm("{\n"
							"   .reg .pred p;               \n"
							"setp.ge.s32        p, %1, 0;   \n"
							"vote.ballot.b32    %0,p;       \n"
							"}"
							: "=r"(mask)
							: "r"(lleafAddr));
						if (!mask)
							break;
					}
					while (lleafAddr < 0)
					{
						for (int triAddr = ~lleafAddr;; triAddr++)
						{
							// Tris in TEX (good to fetch as a single batch)
							const float4 v00 = tex1Dfetch(t_tris, triAddr * 3 + 0 + m_uBVHTriangleOffset);
							const float4 v11 = tex1Dfetch(t_tris, triAddr * 3 + 1 + m_uBVHTriangleOffset);
							const float4 v22 = tex1Dfetch(t_tris, triAddr * 3 + 2 + m_uBVHTriangleOffset);
							unsigned int index = tex1Dfetch(t_triIndices, min(triAddr + m_uBVHIndicesOffset, 8135));

							float Oz = v00.w - lorigx*v00.x - lorigy*v00.y - lorigz*v00.z;
							float invDz = 1.0f / (ldirx*v00.x + ldiry*v00.y + ldirz*v00.z);
							float t = Oz * invDz;

							if (t > ltmin && t < lhitT)
							{
								// Compute and check barycentric u.

								float Ox = v11.w + lorigx*v11.x + lorigy*v11.y + lorigz*v11.z;
								float Dx = ldirx*v11.x + ldiry*v11.y + ldirz*v11.z;
								float u = Ox + t*Dx;

								if (u >= 0.0f)
								{
									// Compute and check barycentric v.

									float Oy = v22.w + lorigx*v22.x + lorigy*v22.y + lorigz*v22.z;
									float Dy = ldirx*v22.x + ldiry*v22.y + ldirz*v22.z;
									float v = Oy + t*Dy;

									if (v >= 0.0f && u + v <= 1.0f)
									{
										// Record intersection.
										// Closest intersection not required => terminate.

										nodeIdx = ~leafAddr;
										lhitT = t;
										hitIndex = (index >> 1) + m_uTriangleOffset;
										bCorrds = Vec2f(u, v);
										if (ANY_HIT)
										{
											nodeAddr = lnodeAddr = EntrypointSentinel;
											break;
										}
									}
								}
							}
							if (index & 1)
								break;
						} // triangle
						hitT = lhitT;

						lleafAddr = lnodeAddr;
						if (lnodeAddr < 0)
						{
							lnodeAddr = *(int*)lstackPtr;
							lstackPtr -= 4;
						}
					}
					//BUGGY
					//if( __popc(__ballot(true)) < DYNAMIC_FETCH_THRESHOLD / 2 )
					//{
					//	//we can't pop yet
					//	nodeAddr = EntrypointSentinel - 1;
					//	//can't break cause we don't want to pop postponed leaf
					//	goto outerlabel;//jump AFTER store cause we will do that later
					//}
				}
				// Another leaf was postponed => process it as well.		
				leafAddr = nodeAddr;
				if (nodeAddr < 0)
				{
					nodeAddr = *(int*)stackPtr;
					stackPtr -= 4;
				}
			} // leaf

            // DYNAMIC FETCH
			//BUGGY
            //if( __popc(__ballot(true)) < DYNAMIC_FETCH_THRESHOLD )
            //    break;
        } // traversal

        // Remap intersected triangle index, and store the result.

		int4 res = make_int4(0,0,0,0);
		if(hitIndex != -1)
		{
			res.x = __float_as_int(hitT);
			res.y = nodeIdx;
			res.z = hitIndex;
			half2 h(bCorrds);
			res.w = *(int*)&h;
		}
		((int4*)a_ResBuffer)[rayidx] = res;*/
//outerlabel: ;
    } while(true);
}

void __internal__IntersectBuffers(int N, traversalRay* a_RayBuffer, traversalResult* a_ResBuffer, bool SKIP_OUTER, bool ANY_HIT)
{
	ThrowCudaErrors(cudaDeviceSetCacheConfig (cudaFuncCachePreferL1));
	unsigned int zero = 0;
	ThrowCudaErrors(cudaMemcpyToSymbol(g_warpCounter, &zero, sizeof(unsigned int)));
	/*if(SKIP_OUTER)
		if(ANY_HIT)
			intersectKernel_SKIPOUTER<true><<< 180, dim3(32, 4, 1)>>>(N, a_RayBuffer, a_ResBuffer);
		else intersectKernel_SKIPOUTER<false><<< 180, dim3(32, 4, 1)>>>(N, a_RayBuffer, a_ResBuffer);
	else*/
	{
		if(ANY_HIT)
			intersectKernel<true><<< 180, dim3(32, 4, 1)>>>(N, a_RayBuffer, a_ResBuffer);
		else intersectKernel<false><<< 180, dim3(32, 4, 1)>>>(N, a_RayBuffer, a_ResBuffer);
	}
	ThrowCudaErrors(cudaDeviceSynchronize());
	g_RayTracedCounterHost += N;
}