#pragma once
#include <vector>
#include "..\e_Mesh.h"

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

BVH_Construction_Result ConstructBVH(const Vec3f* vertices, const unsigned int* indices, unsigned int vCount, unsigned int cCount);

void ConstructBVH(const Vec3f* vertices, const unsigned int* indices, int vCount, int cCount, OutputStream& O, BVH_Construction_Result* out = 0);