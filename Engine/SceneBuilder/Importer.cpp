#include <StdAfx.h>
#include "..\..\Base\FileStream.h"
#include "Importer.h"

void ConstructBVH(float3* vertices, unsigned int* indices, int vCount, int cCount, OutputStream& O)
{	
	bvh_helper::clb c(vCount, cCount, vertices, indices);
	BVHBuilder::BuildBVH(&c, BVHBuilder::Platform());
	O << 5u;
	O << (unsigned long long)c.nodeIndex * 64;
	if(c.nodeIndex)
		O.Write(c.nodes, (unsigned int)c.nodeIndex * sizeof(e_BVHNodeData));
	O << (unsigned long long)c.triIndex * 16;
	if(c.triIndex )
		O.Write(c.tris, (unsigned int)c.triIndex * 16);
	O << (unsigned long long)c.triIndex * 4;
	if(c.triIndex )
		O.Write(c.indices, (unsigned int)c.triIndex * sizeof(int));
	c.Free();
}