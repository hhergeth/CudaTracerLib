#include "TraceHelper.h"
#include <Math/Compression.h>
#include <Math/half.h>
#include "cuda_runtime.h"
#include <Engine/Mesh.h>
#include <Engine/TriangleData.h>
#include <Engine/Material.h>
#include <Engine/TriIntersectorData.h>
#include <SceneTypes/Node.h>
#include <Engine/DynamicScene.h>
#include <Engine/SpatialStructures/BVH/BVHTraversal.h>
#include <Base/Timer.h>
#include "Sampler.h"

namespace CudaTracerLib {

enum
{
	MaxBlockHeight = 6,            // Upper bound for blockDim.y.
	EntrypointSentinel = 0x76543210,   // Bottom-most stack entry, indicating the end of traversal.
};

KernelDynamicScene g_SceneDataDevice;
unsigned int g_RayTracedCounterDevice;
CudaStaticWrapper<SamplerData> g_SamplerDataDevice;

KernelDynamicScene g_SceneDataHost;
unsigned int g_RayTracedCounterHost;
//CudaStaticWrapper  definition in Defines.h, SamplerData in Kernel/Sampler_device.h
CudaStaticWrapper<SamplerData> g_SamplerDataHost;

SamplingSequenceGeneratorHost<IndependantSamplingSequenceGenerator> g_SamplingSequenceGenerator;

texture<float4, 1>		t_nodesA;
texture<float4, 1>		t_tris;
texture<unsigned int,  1>		t_triIndices;
texture<float4, 1>		t_SceneNodes;
texture<float4, 1>		t_NodeTransforms;
texture<float4, 1>		t_NodeInvTransforms;

texture<int2, 1> t_TriDataA;
texture<float4, 1> t_TriDataB;

void traversalResult::toResult(TraceResult* tR, KernelDynamicScene& data) const
{
	tR->m_fDist = dist;
	tR->m_fBaryCoords = ((half2*)&bCoords)->ToFloat2();
	tR->m_nodeIdx = nodeIdx;
	tR->m_triIdx = triIdx;
}

void traversalResult::fromResult(const TraceResult* tR, KernelDynamicScene& data)
{
	half2 X(tR->m_fBaryCoords);
	bCoords = *(int*)&X;
	dist = tR->m_fDist;
	nodeIdx = tR->m_nodeIdx;
	triIdx = tR->m_triIdx;
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

template<bool USE_ALPHA> CUDA_FUNC_IN bool __traceRay_internal__(const Vec3f& dir, const Vec3f& ori, TraceResult* a_Result)
{
	return TracerayTemplate(Ray(ori, dir), a_Result->m_fDist, [&](int nodeIdx)
	{
		Node* N = g_SceneData.m_sNodeData.Data + nodeIdx;
		KernelMesh mesh = g_SceneData.m_sMeshData[N->m_uMeshIndex];
		unsigned int meshBvhTriOff = mesh.m_uBVHTriangleOffset, meshBvhIndOff = mesh.m_uBVHIndicesOffset, meshTriOff = mesh.m_uTriangleOffset;
		unsigned int nodeMatOff = N->m_uMaterialOffset;
		float4x4 modl;
		loadInvModl(nodeIdx, &modl);
		Vec3f d = modl.TransformDirection(dir), o = modl.TransformPoint(ori);
		return TracerayTemplate(Ray(o, d), a_Result->m_fDist, [&](int triIdx)
		{
			bool found = false;
			for (int triAddr = triIdx;; triAddr++)
			{
#ifdef ISCUDA
				const float4 v00 = tex1Dfetch(t_tris, meshBvhTriOff + triAddr * 3 + 0);
				const float4 v11 = tex1Dfetch(t_tris, meshBvhTriOff + triAddr * 3 + 1);
				const float4 v22 = tex1Dfetch(t_tris, meshBvhTriOff + triAddr * 3 + 2);
				unsigned int index = tex1Dfetch(t_triIndices, meshBvhIndOff + triAddr);
#else
				Vec4f* dat = (Vec4f*)g_SceneData.m_sBVHIntData.Data;
				const Vec4f v00 = dat[meshBvhTriOff + triAddr * 3 + 0];
				const Vec4f v11 = dat[meshBvhTriOff + triAddr * 3 + 1];
				const Vec4f v22 = dat[meshBvhTriOff + triAddr * 3 + 2];
				unsigned int index = g_SceneData.m_sBVHIndexData.Data[meshBvhIndOff + triAddr].index;
#endif

				float Oz = v00.w - o.x*v00.x - o.y*v00.y - o.z*v00.z;
				float invDz = 1.0f / (d.x*v00.x + d.y*v00.y + d.z*v00.z);
				float t = Oz * invDz;
				if (t > 1e-2f && t < a_Result->m_fDist)
				{
					float Ox = v11.w + o.x*v11.x + o.y*v11.y + o.z*v11.z;
					float Dx = d.x*v11.x + d.y*v11.y + d.z*v11.z;
					float u = Ox + t*Dx;
					if (u >= 0.0f)
					{
						float Oy = v22.w + o.x*v22.x + o.y*v22.y + o.z*v22.z;
						float Dy = d.x*v22.x + d.y*v22.y + d.z*v22.z;
						float v = Oy + t*Dy;
						if (v >= 0.0f && u + v <= 1.0f)
						{
							unsigned int ti = index >> 1;

							bool alphaSurvive = true;
							if (USE_ALPHA)
							{
								TriangleData* tri = g_SceneData.m_sTriData.Data + ti + meshTriOff;
								unsigned int mIdx = tri->getMatIndex(nodeMatOff);
								auto& mat = g_SceneData.m_sMatData[mIdx];
								if (mat.AlphaMap.used())
								{
#ifdef ISCUDA
									float4 rowC = tex1Dfetch(t_TriDataB, ti * 4 + 2);
									float4 rowD = tex1Dfetch(t_TriDataB, ti * 4 + 3);
									Vec2f b = Vec2f(rowC.z, rowC.w), a = Vec2f(rowD.x, rowD.y), c = Vec2f(rowD.z, rowD.w);
#else
									Vec2f a, b, c;
									tri->getUVSetData(0, a, b, c);
#endif
									Vec2f uv = u * a + v * b + (1 - u - v) * c;
									alphaSurvive = mat.AlphaTest(Vec2f(u, v), uv);
								}
							}
							if (alphaSurvive)
							{
								a_Result->m_nodeIdx = nodeIdx;
								a_Result->m_triIdx = ti + meshTriOff;
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
			return found;
		}, t_nodesA, g_SceneData.m_sBVHNodeData.Data, mesh.m_uBVHNodeOffset, 0);
	}, t_SceneNodes, g_SceneData.m_sSceneBVH.m_pNodes, 0, g_SceneData.m_sSceneBVH.m_sStartNode);
}

bool traceRay(const Vec3f& dir, const Vec3f& ori, TraceResult* a_Result)
{
	Platform::Increment(&g_RayTracedCounter);
	if(!g_SceneData.m_sNodeData.UsedCount)
		return false;
	return g_SceneData.doAlphaMapping ? __traceRay_internal__<true>(dir, ori, a_Result) : __traceRay_internal__<false>(dir, ori, a_Result);
}

void UpdateKernel(DynamicScene* a_Scene, ISamplingSequenceGenerator& sampler)
{
	GenerateNewRandomSequences(sampler);

	if (!a_Scene)
		return;
	KernelDynamicScene a_Data = a_Scene->getKernelSceneData();

	size_t offset;
	cudaChannelFormatDesc	cdf4 = cudaCreateChannelDesc<float4>(),
							cdu1 = cudaCreateChannelDesc<unsigned int>(),
							cdi2 = cudaCreateChannelDesc<int2>(),
							cdh4 = cudaCreateChannelDescHalf4();
	ThrowCudaErrors(cudaBindTexture(&offset, &t_nodesA, a_Data.m_sBVHNodeData.Data, &cdf4, a_Data.m_sBVHNodeData.UsedCount * sizeof(BVHNodeData)));
	ThrowCudaErrors(cudaBindTexture(&offset, &t_tris, a_Data.m_sBVHIntData.Data, &cdf4, a_Data.m_sBVHIntData.UsedCount * sizeof(TriIntersectorData)));
	ThrowCudaErrors(cudaBindTexture(&offset, &t_triIndices, a_Data.m_sBVHIndexData.Data, &cdu1, a_Data.m_sBVHIndexData.UsedCount * sizeof(TriIntersectorData2)));
	ThrowCudaErrors(cudaBindTexture(&offset, &t_SceneNodes, a_Data.m_sSceneBVH.m_pNodes, &cdf4, a_Data.m_sSceneBVH.m_uNumNodes * sizeof(BVHNodeData)));
	ThrowCudaErrors(cudaBindTexture(&offset, &t_NodeTransforms, a_Data.m_sSceneBVH.m_pNodeTransforms, &cdf4, a_Data.m_sNodeData.UsedCount * sizeof(float4x4)));
	ThrowCudaErrors(cudaBindTexture(&offset, &t_NodeInvTransforms, a_Data.m_sSceneBVH.m_pInvNodeTransforms, &cdf4, a_Data.m_sNodeData.UsedCount * sizeof(float4x4)));
#ifdef EXT_TRI
	ThrowCudaErrors(cudaBindTexture(&offset, &t_TriDataA, a_Data.m_sTriData.Data, &cdi2, a_Data.m_sTriData.UsedCount * sizeof(TriangleData)));
	ThrowCudaErrors(cudaBindTexture(&offset, &t_TriDataB, a_Data.m_sTriData.Data, &cdh4, a_Data.m_sTriData.UsedCount * sizeof(TriangleData)));
#endif

	unsigned int b = 0;
	void* symAdd = 0;
	ThrowCudaErrors(cudaGetSymbolAddress(&symAdd, g_RayTracedCounterDevice));
	if (symAdd)
		ThrowCudaErrors(cudaMemcpyToSymbol(g_RayTracedCounterDevice, &b, sizeof(b)));
	ThrowCudaErrors(cudaGetSymbolAddress(&symAdd, g_SceneDataDevice));
	if (symAdd)
		ThrowCudaErrors(cudaMemcpyToSymbol(g_SceneDataDevice, &a_Data, sizeof(a_Data)));

	g_SceneDataHost = a_Scene->getKernelSceneData(false);
	g_RayTracedCounterHost = 0;
}

void UpdateKernel(DynamicScene* a_Scene)
{
	UpdateKernel(a_Scene, g_SamplingSequenceGenerator);
}

void UpdateSamplerData(unsigned int num_sequences, unsigned int sequence_length)
{
	struct helper
	{
		static bool is_changed(const RandomSamplerData& data, unsigned int num_sequences, unsigned int sequence_length)
		{
			return data.getNumSequences() != num_sequences;
		}

		static bool is_changed(const SequenceSamplerData& data, unsigned int num_sequences, unsigned int sequence_length)
		{
			return data.getSequenceLength() != sequence_length || data.getNumSequences() != num_sequences;
		}
	};

	if (!helper::is_changed(g_SamplerData, num_sequences, sequence_length))
		return;

	//create the buffer obj
	g_SamplerDataHost->Free();
	new(g_SamplerDataHost.operator->()) SamplerData(num_sequences, sequence_length);

	//copy obj to device
	void* symAdd;
	ThrowCudaErrors(cudaGetSymbolAddress(&symAdd, g_SamplerDataDevice));
	if (symAdd)
		ThrowCudaErrors(cudaMemcpyToSymbol(g_SamplerDataDevice, &g_SamplerDataHost, sizeof(g_SamplerDataHost)));
}

void InitializeKernel()
{
	new(g_SamplerDataHost.operator->()) SamplerData(1, 1);
	UpdateSamplerData(1 << 12, 30);
}

void GenerateNewRandomSequences(ISamplingSequenceGenerator& sampler)
{
	sampler.Compute(g_SamplerDataHost);
}

void GenerateNewRandomSequences()
{
	GenerateNewRandomSequences(g_SamplingSequenceGenerator);
}

void DeinitializeKernel()
{
	g_SamplerDataHost->Free();
}

void fillDG(const Vec2f& bary, unsigned int triIdx, unsigned int nodeIdx, DifferentialGeometry& dg)
{
	float4x4 localToWorld, worldToLocal;
	loadModl(nodeIdx, &localToWorld);
	loadInvModl(nodeIdx, &worldToLocal);
	dg.bary = bary;
	dg.hasUVPartials = false;
#if defined(ISCUDA) && NUM_UV_SETS == 1 && defined(EXT_TRI)
	int2 nme = tex1Dfetch(t_TriDataA, triIdx * 4 + 0);
	float4 rowB = tex1Dfetch(t_TriDataB, triIdx * 4 + 1);
	float4 rowC = tex1Dfetch(t_TriDataB, triIdx * 4 + 2);
	float4 rowD = tex1Dfetch(t_TriDataB, triIdx * 4 + 3);
	NormalizedT<Vec3f> na = Uchar2ToNormalizedFloat3(nme.x), nb = Uchar2ToNormalizedFloat3(nme.x >> 16), nc = Uchar2ToNormalizedFloat3(nme.y);
	float w = 1.0f - dg.bary.x - dg.bary.y, u = dg.bary.x, v = dg.bary.y;
	dg.extraData = nme.y >> 24;
	Vec3f n = normalize(u * na + v * nb + w * nc);
	Vec3f dpdu = Vec3f(rowB.x, rowB.y, rowB.z);
	Vec3f dpdv = Vec3f(rowB.w, rowC.x, rowC.y);
	Vec3f s = dpdu - n * dot(n, dpdu);
	Vec3f t = cross(s, n);
	s = localToWorld.TransformDirection(s); t = localToWorld.TransformDirection(t);
	dg.sys = Frame(s.normalized(), t.normalized(), cross(t, s).normalized());
	dg.dpdu = localToWorld.TransformDirection(dpdu);
	dg.dpdv = localToWorld.TransformDirection(dpdv);
	dg.n = cross(dg.dpdu, dg.dpdv).normalized();
	Vec2f ta = Vec2f(rowC.z, rowC.w), tb = Vec2f(rowD.x, rowD.y), tc = Vec2f(rowD.z, rowD.w);
	dg.uv[0] = u * ta + v * tb + w * tc;

	if (dot(dg.n, dg.sys.n) < 0.0f)
		dg.n = -dg.n;
#else
	g_SceneData.m_sTriData[triIdx].fillDG(localToWorld, dg);
#endif
}

unsigned int k_getNumRaysTraced()
{
	unsigned int i;
	ThrowCudaErrors(cudaMemcpyFromSymbol(&i, g_RayTracedCounterDevice, sizeof(unsigned int)));
	return i + g_RayTracedCounterHost;
}

void k_setNumRaysTraced(unsigned int i)
{
	g_RayTracedCounterHost = i;
	ThrowCudaErrors(cudaMemcpyToSymbol(g_RayTracedCounterDevice, &i, sizeof(unsigned int)));
}

#define DYNAMIC_FETCH_THRESHOLD 20
#define STACK_SIZE 32
__device__ int g_warpCounter;

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
	int nodeIdx;

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
			nodeIdx = -1;
		}

		// Traversal loop.
		/*TraceResult r2 = traceRay(Ray(a_RayBuffer[rayidx].a.getXYZ(), a_RayBuffer[rayidx].b.getXYZ()));
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
		nodeAddr = EntrypointSentinel;*/

		if (g_SceneData.m_sNodeData.UsedCount == 0)
			nodeAddr = EntrypointSentinel;

		while (nodeAddr != EntrypointSentinel)
		{
			// Traverse internal nodes until all SIMD lanes have found a leaf.
			while (((unsigned int)nodeAddr) < ((unsigned int)EntrypointSentinel))   // functionally equivalent, but faster
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
				const float c0min = kepler_math::spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, tmin);
				const float c0max = kepler_math::spanEndKepler  (c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hitT);
				const float c1lox = n1xy.x * idirx - oodx;
				const float c1hix = n1xy.y * idirx - oodx;
				const float c1loy = n1xy.z * idiry - oody;
				const float c1hiy = n1xy.w * idiry - oody;
				const float c1min = kepler_math::spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, tmin);
				const float c1max = kepler_math::spanEndKepler  (c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, hitT);

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
				Node* N = g_SceneData.m_sNodeData.Data + (~leafAddr);
				//if (terminated)
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
					while (((unsigned int)lnodeAddr) < ((unsigned int)EntrypointSentinel))
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
						const float c0min = kepler_math::spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, ltmin);
						const float c0max = kepler_math::spanEndKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, lhitT);
						const float c1lox = n1xy.x * lidirx - loodx;
						const float c1hix = n1xy.y * lidirx - loodx;
						const float c1loy = n1xy.z * lidiry - loody;
						const float c1hiy = n1xy.w * lidiry - loody;
						const float c1min = kepler_math::spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, ltmin);
						const float c1max = kepler_math::spanEndKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, lhitT);

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
							unsigned int index = tex1Dfetch(t_triIndices, triAddr + m_uBVHIndicesOffset);

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

		uint4 res = make_uint4(0, UINT_MAX, UINT_MAX,0);
		if(hitIndex != -1)
		{
			res.x = __float_as_int(hitT);
			res.y = nodeIdx;
			res.z = hitIndex;
			half2 h(bCorrds);
			res.w = *(int*)&h;
		}
		((uint4*)a_ResBuffer)[rayidx] = res;
//outerlabel: ;
	} while(true);
}

void __internal__IntersectBuffers(int N, traversalRay* a_RayBuffer, traversalResult* a_ResBuffer, bool SKIP_OUTER, bool ANY_HIT)
{
	ThrowCudaErrors(cudaDeviceSetCacheConfig (cudaFuncCachePreferL1));
	unsigned int zero = 0;
	ThrowCudaErrors(cudaMemcpyToSymbol(g_warpCounter, &zero, sizeof(unsigned int)));
	if(ANY_HIT)
		intersectKernel<true><<< 180, dim3(32, 4, 1)>>>(N, a_RayBuffer, a_ResBuffer);
	else intersectKernel<false><<< 180, dim3(32, 4, 1)>>>(N, a_RayBuffer, a_ResBuffer);
	ThrowCudaErrors(cudaDeviceSynchronize());
	g_RayTracedCounterHost += N;
}

}
