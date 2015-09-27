#pragma once

#include "../Defines.h"
#include "../MathTypes.h"
#include "../Engine/e_TriIntersectorData.h"

template<typename CLB> CUDA_FUNC_IN bool k_TraceRayTemplate(const Ray& r, float& rayT, const CLB& clb, texture<float4, 1> bvhNodes_texture, e_BVHNodeData* bvhNodes, int bvhNodesOffset = 0, int startNode = 0)
{
	if (startNode < 0)
		return clb(~startNode);
	bool found = false;
	int traversalStack[64];
	traversalStack[0] = EntrypointSentinel;
	float   dirx = r.direction.x;
	float   diry = r.direction.y;
	float   dirz = r.direction.z;
	const float ooeps = math::exp2(-80.0f);
	float   idirx = 1.0f / (math::abs(r.direction.x) > ooeps ? r.direction.x : copysignf(ooeps, r.direction.x));
	float   idiry = 1.0f / (math::abs(r.direction.y) > ooeps ? r.direction.y : copysignf(ooeps, r.direction.y));
	float   idirz = 1.0f / (math::abs(r.direction.z) > ooeps ? r.direction.z : copysignf(ooeps, r.direction.z));
	float   origx = r.origin.x;
	float	origy = r.origin.y;
	float	origz = r.origin.z;						// Ray origin.
	float   oodx = origx * idirx;
	float   oody = origy * idiry;
	float   oodz = origz * idirz;
	char*   stackPtr;                       // Current position in traversal stack.
	int     leafAddr;                       // First postponed leaf, non-negative if none.
	int     nodeAddr = EntrypointSentinel;  // Non-negative: current internal node, negative: second postponed leaf.
	stackPtr = (char*)&traversalStack[0];
	leafAddr = 0;   // No postponed leaf.
	nodeAddr = startNode;   // Start from the root.
	while (nodeAddr != EntrypointSentinel)
	{
		while (unsigned int(nodeAddr) < unsigned int(EntrypointSentinel))
		{
#ifdef ISCUDA
			const float4 n0xy = tex1Dfetch(bvhNodes_texture, bvhNodesOffset + nodeAddr + 0); // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
			const float4 n1xy = tex1Dfetch(bvhNodes_texture, bvhNodesOffset + nodeAddr + 1); // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
			const float4 nz = tex1Dfetch(bvhNodes_texture, bvhNodesOffset + nodeAddr + 2); // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
			float4 tmp = tex1Dfetch(bvhNodes_texture, bvhNodesOffset + nodeAddr + 3); // child_index0, child_index1
#else
			Vec4f* dat = (Vec4f*)bvhNodes;
			const Vec4f n0xy = dat[bvhNodesOffset + nodeAddr + 0];
			const Vec4f n1xy = dat[bvhNodesOffset + nodeAddr + 1];
			const Vec4f nz = dat[bvhNodesOffset + nodeAddr + 2];
			Vec4f tmp = dat[bvhNodesOffset + nodeAddr + 3];
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
			const float c0max = math::spanEndKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, rayT);
			const float c1lox = n1xy.x * idirx - oodx;
			const float c1hix = n1xy.y * idirx - oodx;
			const float c1loy = n1xy.z * idiry - oody;
			const float c1hiy = n1xy.w * idiry - oody;
			const float c1min = math::spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, 0);
			const float c1max = math::spanEndKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, rayT);
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

			if (nodeAddr < 0 && leafAddr >= 0)     // Postpone max 1
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
			if (!mask)
				break;
		}
		while (leafAddr < 0)
		{
			if (leafAddr != -214783648)
				found |= clb(~leafAddr);
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
