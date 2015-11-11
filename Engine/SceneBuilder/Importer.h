#pragma once
#include <vector>
#include <Engine/e_Mesh.h>

struct BVH_Construction_Result
{
	std::vector<e_BVHNodeData> nodes;
	std::vector<e_TriIntersectorData> tris;
	std::vector<e_TriIntersectorData2> tris2;
	AABB box;
};

void ConstructBVH(const Vec3f* vertices, const unsigned int* indices, unsigned int vCount, unsigned int cCount, BVH_Construction_Result& res);

void ConstructBVH(const Vec3f* vertices, const unsigned int* indices, int vCount, int cCount, FileOutputStream& O, BVH_Construction_Result* out = 0);