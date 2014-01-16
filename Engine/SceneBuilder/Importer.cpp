#include <StdAfx.h>
#include "..\..\Base\FileStream.h"
#include "Importer.h"

void ConstructBVH(float3* vertices, unsigned int* indices, int vCount, int cCount, OutputStream& O, BVH_Construction_Result* out)
{	
	bvh_helper::clb c(vCount, cCount, vertices, indices);
	BVHBuilder::BuildBVH(&c, BVHBuilder::Platform());
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