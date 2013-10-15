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
		unsigned int* I;
		unsigned int v, i;
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
			: V(_V), I(_I), v(_v), i(_i)
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
				out->Enlarge(V[I[index * 3 + i]]);
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
			t->setData(V[I[pNode * 3 + 0]], V[I[pNode * 3 + 1]], V[I[pNode * 3 + 2]]);
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

inline void ConstructBVH(float3* vertices, unsigned int* indices, int vCount, int cCount, e_BVHNodeData** nodes = 0, float4** tris = 0, unsigned int** ind = 0, int* A = 0, int* B = 0)
{
	bvh_helper::clb c(vCount, cCount, vertices, indices);
	BVHBuilder::BuildBVH(&c, BVHBuilder::Platform());
	if(nodes)
		*nodes = c.nodes;
	else delete [] c.nodes;
	if(tris)
		*tris = c.tris;
	else delete [] c.tris;
	if(ind)
		*ind = c.indices;
	else delete [] c.indices;
	if(A)
		*A = c.nodeIndex;
	if(B)
		*B = c.triIndex;

}

void ConstructBVH(float3* vertices, unsigned int* indices, int vCount, int cCount, OutputStream& O);