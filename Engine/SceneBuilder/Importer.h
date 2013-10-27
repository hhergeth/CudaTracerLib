#pragma once
#include <vector>
#include "..\..\Base\BVHBuilder.h"
#include "..\e_Mesh.h"
/*
void ConstructBVH(FW::Mesh<FW::VertexP>& M, FW::OutputStream& O, float4** a_Nodes);

void exportBVH(char* Input, char* Output);

*/
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
		float4* tris;
		unsigned int triIndex;
		unsigned int* indices;
		AABB box;
		int startNode;
		int L0, L1;
	public:
		clb(int _v, int _i, float3* _V, unsigned int* _I)
			: V(_V), Iab(_I), v(_v), i(_i)
		{
			nodeIndex = triIndex = 0;
			const float duplicates = 0.35f, f = 1.0f + duplicates;
			L0 = (int)(float(_i / 3) * f);
			L1 = (int)(float(_i / 3) * f * 4);
			nodes = new e_BVHNodeData[L0];
			tris = new float4[L1];
			indices = new unsigned int[L1];
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
			unsigned int c = triIndex;
			triIndex += 3;
			if(triIndex > L1)
				throw 1;
			e_TriIntersectorData* t = (e_TriIntersectorData*)(tris + c);
			t->setData(V[_index(pNode, 0)], V[_index(pNode, 1)], V[_index(pNode, 2)]);
			indices[c] = pNode;
			return c;
		}
		virtual void handleLastLeafObject()
		{
			*(int*)&tris[triIndex].x = 0x80000000;
			indices[triIndex] = -1;
			triIndex += 1;
			if(triIndex > L1)
				throw 1;
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
	float4* tris;
	unsigned int* ind;
	unsigned int nodeCount;
	//number of float4's <=> indices
	unsigned int triCount;
	AABB box;
	void Free()
	{
		delete [] nodes;
		delete [] tris;
		delete [] ind;
	}
};

inline BVH_Construction_Result ConstructBVH(float3* vertices, unsigned int* indices, unsigned int vCount, unsigned int cCount)
{
	bvh_helper::clb c(vCount, cCount, vertices, indices);
	BVHBuilder::BuildBVH(&c, BVHBuilder::Platform());
	BVH_Construction_Result r;
	r.box = c.box;
	r.ind = c.indices;
	r.nodeCount = c.nodeIndex;
	r.nodes = c.nodes;
	r.triCount = c.triIndex;
	r.tris = c.tris;
	return r;
}

void ConstructBVH(float3* vertices, unsigned int* indices, int vCount, int cCount, OutputStream& O);