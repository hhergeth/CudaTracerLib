#pragma once
#include <vector>
#include "..\..\Base\BVHBuilder.h"
#include "..\e_Mesh.h"
/*
void ConstructBVH(FW::Mesh<FW::VertexP>& M, FW::OutputStream& O, float4** a_Nodes);

void exportBVH(char* Input, char* Output);

void ConstructBVH2(FW::MeshBase* M, FW::OutputStream& O);
*/
inline void ConstructBVH(float3* vertices, unsigned int* indices, int vCount, int cCount, std::vector<e_BVHNodeData>* nodes = 0, std::vector<char>* tris = 0, std::vector<int>* indicesA = 0)
{
	class clb : public IBVHBuilderCallback
	{
		float3* V;
		unsigned int* I;
		unsigned int v, i;
	public:
		std::vector<e_BVHNodeData> nodes;
		std::vector<char> tris;
		std::vector<int> indices;
		AABB box;
		int startNode;
	public:
		clb(int _v, int _i, float3* _V, unsigned int* _I)
			: V(_V), I(_I), v(_v), i(_i)
		{
			nodes.reserve(_i / 3 / 2);
			tris.reserve(12 * _i / 3);
			indices.reserve(_i / 3);
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
			*index = (int)nodes.size();
			nodes.push_back(e_BVHNodeData());
			return &nodes[*index];
		}
		virtual unsigned int handleLeafObjects(unsigned int pNode)
		{
			unsigned int c = (unsigned int)tris.size() / 4;
			tris.resize(tris.size() + 48);
			e_TriIntersectorData* t = (e_TriIntersectorData*)&tris[tris.size() - 48];
			t->setData(V[I[pNode * 3 + 0]], V[I[pNode * 3 + 1]], V[I[pNode * 3 + 2]]);
			indices.resize(indices.size() + 3);
			*(int*)&indices[indices.size() - 3] = pNode;
			return c;
		}
		virtual void handleLastLeafObject()
		{
			tris.resize(tris.size() + 16);
			e_TriIntersectorData* t = (e_TriIntersectorData*)&tris[tris.size() - 16];
			t->a.x = 0x80000000;
			indices.resize(indices.size() + 1);
			*(int*)&indices[indices.size() - 1] = -1;
		}
		virtual void HandleStartNode(int startNode)
		{
			this->startNode = startNode;
		}
	};
	clb c(vCount, cCount, vertices, indices);
	BVHBuilder::BuildBVH(&c, BVHBuilder::Platform(0x7FFFFFF));
	if(nodes)
		*nodes = c.nodes;
	if(tris)
		*tris = c.tris;
	if(indicesA)
		*indicesA = c.indices;
}

inline void ConstructBVH(float3* vertices, unsigned int* indices, int vCount, int cCount, OutputStream& O)
{
	std::vector<e_BVHNodeData> nodes;
	std::vector<char> tris;
	std::vector<int> ind;
	ConstructBVH(vertices, indices, vCount, cCount, &nodes, &tris, &ind);
	O << 5u;
	O << (unsigned long long)nodes.size() * 64;
	if(nodes.size())
		O.Write(&nodes[0], (unsigned int)nodes.size() * sizeof(e_BVHNodeData));
	O << (unsigned long long)tris.size();
	if(tris.size())
		O.Write(&tris[0], (unsigned int)tris.size());
	O << (unsigned long long)ind.size() * 4;
	if(ind.size())
		O.Write(&ind[0], (unsigned int)ind.size() * sizeof(int));
}