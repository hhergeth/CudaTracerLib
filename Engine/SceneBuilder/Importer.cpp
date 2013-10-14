#include "StdAfx.h"/*
#include "CudaBVH.hpp"

using namespace FW;

void ConstructBVH(Mesh<VertexP>& M, OutputStream& O, float4** a_Nodes)
{
	Scene S(M);
	BVH B(&S, Platform(), BVH::BuildParams());
	CudaBVH B2(B, BVHLayout_Compact2);
	*a_Nodes = new float4[B2.getNodeBuffer().getSize() / 16];
	memcpy(*a_Nodes, B2.getNodeBuffer().getPtr(),B2.getNodeBuffer().getSize()); 
	B2.serialize(O);
}

#include <io/File.hpp>
void exportBVH(char* Input, char* Output)
{
	FW::BVH::BuildParams m_buildParams;
	FW::BVH::Stats stats;
    m_buildParams.stats = &stats;

	FW::MeshBase* mesh = importMesh(Input);
	FW::Scene* m_scene = new Scene(*mesh);

	FW::printf("\nBuilding BVH...\nThis will take a while.\n");

    // Build BVH.

    FW::BVH bvh(m_scene, FW::Platform(), m_buildParams);
    stats.print();
	FW::CudaBVH* m_bvh = new FW::CudaBVH(bvh, BVHLayout_Compact2);

    // Display status.

    FW::printf("Done.\n\n");
}

#include "stdafx.h"
#include <iostream>
#include "..\..\Base\Timer.h"
void ConstructBVH2(FW::MeshBase* M, FW::OutputStream& O)
{
	cTimer T;
	T.StartTimer();
	FW::BVH::BuildParams m_buildParams;
	FW::BVH::Stats stats;
	FW::Scene* m_scene = new FW::Scene(*M);
    FW::BVH bvh(m_scene, FW::Platform(), m_buildParams);
    stats.print();
	FW::CudaBVH* m_bvh = new FW::CudaBVH(bvh, BVHLayout_Compact2);
	m_bvh->serialize(O);
	double tSec = T.EndTimer();
	std::cout << "BVH Construction of " << M->numTriangles() << " objects took " << tSec << " seconds\n";
}
*/

#include "..\..\Base\BVHBuilder.h"
#include "..\e_Mesh.h"
#include <vector>

void ConstructBVH(float3* vertices, unsigned int* indices, int vCount, int cCount, std::vector<e_BVHNodeData>* nodes, std::vector<char>* tris, std::vector<int>* indicesA)
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
			*index = nodes.size();
			nodes.push_back(e_BVHNodeData());
			return &nodes[*index];
		}
		virtual unsigned int handleLeafObjects(unsigned int pNode)
		{
			unsigned int c = tris.size() / 4;
			e_TriIntersectorData* t = (e_TriIntersectorData*)tris.get_allocator().allocate(48);
			t->setData(V[I[pNode * 3 + 0]], V[I[pNode * 3 + 1]], V[I[pNode * 3 + 2]]);
			*indices.get_allocator().allocate(12) = pNode;
			return c;
		}
		virtual void handleLastLeafObject()
		{
			e_TriIntersectorData* t = (e_TriIntersectorData*)tris.get_allocator().allocate(4);
			t->a.x = 0x80000000;
			*indices.get_allocator().allocate(4) = -1;
		}
		virtual void HandleStartNode(int startNode)
		{
			this->startNode = startNode;
		}
	};
	clb c(vCount, cCount, vertices, indices);
	BVHBuilder::BuildBVH(&c, BVHBuilder::Platform(0x7FFFFFF));
	*nodes = c.nodes;
	*tris = c.tris;
	*indicesA = c.indices;
}

void ConstructBVH(float3* vertices, unsigned int* indices, int vCount, int cCount, OutputStream& O)
{
	std::vector<e_BVHNodeData> nodes;
	std::vector<char> tris;
	std::vector<int> ind;
	ConstructBVH(vertices, indices, vCount, cCount, &nodes, &tris, &ind);
	O << 5u;
	O.Write(&nodes[0], nodes.size());
	O.Write(&tris[0], tris.size());
	O.Write(&ind[0], ind.size());
}