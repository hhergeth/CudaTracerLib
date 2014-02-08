#pragma once
#include <vector>
#include "..\..\Base\BVHBuilder.h"
#include "..\e_Mesh.h"

namespace bvh_helper
{
	class clb : public IBVHBuilderCallback
	{
		float3* V;
		unsigned int* Iab;
		unsigned int v, i;
		unsigned int _index(unsigned int i, unsigned int o) const
		{
			return Iab ? Iab[i * 3 + o] : (i * 3 + o);
		}
	public:
		e_BVHNodeData* nodes;
		unsigned int nodeIndex;
		e_TriIntersectorData* tris;
		unsigned int triIndex;
		e_TriIntersectorData2* indices;
		AABB box;
		int startNode;
		unsigned int L0, L1;
	public:
		clb(unsigned int _v, unsigned int _i, float3* _V, unsigned int* _I)
			: V(_V), Iab(_I), v(_v), i(_i)
		{
			std::cout << _v << "\n";
			std::cout << _i << "\n";
			nodeIndex = triIndex = 0;
			const float duplicates = 0.5f, f = 1.0f + duplicates;
			L0 = (int)(float(_i / 3) * f);
			L1 = (int)(float(_i / 3) * f * 4);
			nodes = new e_BVHNodeData[L0];
			tris = new e_TriIntersectorData[L1];
			indices = new e_TriIntersectorData2[L1];
			Platform::SetMemory(indices, L1 * sizeof(e_TriIntersectorData2));
		}
		void Free()
		{
			delete [] nodes;
			delete [] tris;
			delete [] indices;
		}
		virtual unsigned int Count() const
		{
			return i / 3;
		}
		virtual void getBox(unsigned int index, AABB* out) const
		{
			*out = AABB::Identity();
			for(int i = 0; i < 3; i++)
				out->Enlarge(V[_index(index, i)]);
			if(fminf(out->Size()) < 0.01f)
				out->maxV += make_float3(0.01f);
		}
		virtual void HandleBoundingBox(const AABB& box)
		{
			this->box = box;
		}
		virtual e_BVHNodeData* HandleNodeAllocation(int* index)
		{
			*index = nodeIndex++ * 4;
			if(nodeIndex > L0)
				throw 1;
			return nodes + *index / 4;
		}
		virtual unsigned int handleLeafObjects(unsigned int pNode)
		{
			unsigned int c = triIndex++;
			if(triIndex > L1)
				throw 1;
			indices[c].setIndex(pNode);
			tris[c].setData(V[_index(pNode, 0)], V[_index(pNode, 1)], V[_index(pNode, 2)]);
			indices[c].setFlag(false);
			indices[c].setIndex(pNode);
			return c;
		}
		virtual void handleLastLeafObject()
		{
			//*(int*)&tris[triIndex].x = 0x80000000;
			//indices[triIndex] = -1;
			//triIndex += 1;
			//if(triIndex > L1)
			//	throw 1;
			indices[triIndex - 1].setFlag(true);
		}
		virtual void HandleStartNode(int startNode)
		{
			this->startNode = startNode;
		}
	};
}

struct BVH_Construction_Result
{
	e_BVHNodeData* nodes;
	e_TriIntersectorData* tris;
	e_TriIntersectorData2* tris2;
	unsigned int nodeCount;
	unsigned int triCount;
	AABB box;
	void Free()
	{
		delete [] nodes;
		delete [] tris;
		delete [] tris2;
	}
};

inline BVH_Construction_Result ConstructBVH(float3* vertices, unsigned int* indices, unsigned int vCount, unsigned int cCount)
{
	bvh_helper::clb c(vCount, cCount, vertices, indices);
	BVHBuilder::BuildBVH(&c, BVHBuilder::Platform());
	BVH_Construction_Result r;
	r.box = c.box;
	r.tris2 = c.indices;
	r.nodeCount = c.nodeIndex;
	r.nodes = c.nodes;
	r.triCount = c.triIndex;
	r.tris = c.tris;
	return r;
}

void ConstructBVH(float3* vertices, unsigned int* indices, int vCount, int cCount, OutputStream& O, BVH_Construction_Result* out = 0);