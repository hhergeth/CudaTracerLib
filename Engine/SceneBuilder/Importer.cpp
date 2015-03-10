#include <StdAfx.h>
#include "..\..\Base\FileStream.h"
#include "Importer.h"
#include "SplitBVHBuilder.hpp"

BVH_Construction_Result ConstructBVH(const Vec3f* vertices, const unsigned int* indices, unsigned int vCount, unsigned int cCount)
{
	bvh_helper::clb c(vCount, cCount, vertices, indices);
	//BVHBuilder::BuildBVH(&c, BVHBuilder::Platform());
	SplitBVHBuilder bu(&c, BVHBuilder::Platform(), BVHBuilder::BuildParams()); bu.run();
	BVH_Construction_Result r;
	r.box = c.box;
	r.tris2 = c.indices;
	r.nodeCount = c.nodeIndex;
	r.nodes = c.nodes;
	r.triCount = c.triIndex;
	r.tris = c.tris;
	return r;
}

void ConstructBVH(const Vec3f* vertices, const unsigned int* indices, int vCount, int cCount, OutputStream& O, BVH_Construction_Result* out)
{	
	bvh_helper::clb c(vCount, cCount, vertices, indices);
	//BVHBuilder::BuildBVH(&c, BVHBuilder::Platform());
	SplitBVHBuilder bu(&c, BVHBuilder::Platform(), BVHBuilder::BuildParams()); bu.run();
	O << (unsigned long long)c.nodeIndex;
	if(c.nodeIndex)
		O.Write(c.nodes, (unsigned int)c.nodeIndex * sizeof(e_BVHNodeData));
	O << (unsigned long long)c.triIndex;
	if(c.triIndex )
		O.Write(c.tris, (unsigned int)c.triIndex * sizeof(e_TriIntersectorData));
	O << (unsigned long long)c.triIndex;
	if(c.triIndex )
		O.Write(c.indices, (unsigned int)c.triIndex * sizeof(e_TriIntersectorData2));
	if(out)
	{
		out->box = c.box;
		out->tris2 = c.indices;
		out->nodeCount = c.nodeIndex;
		out->nodes = c.nodes;
		out->triCount = c.triIndex;
		out->tris = c.tris;
	}
	else c.Free();
}