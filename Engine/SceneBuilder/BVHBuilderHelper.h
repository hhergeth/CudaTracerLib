#pragma once
#include <vector>
#include <Engine/Mesh.h>

namespace CudaTracerLib {

struct BVH_Construction_Result
{
	std::vector<BVHNodeData> nodes;
	std::vector<TriIntersectorData> tris;
	std::vector<TriIntersectorData2> tris2;
	AABB box;
};

CTL_EXPORT void ConstructBVH(const Vec3f* vertices, const unsigned int* indices, unsigned int vCount, unsigned int cCount, BVH_Construction_Result& res);

CTL_EXPORT void ConstructBVH(const Vec3f* vertices, const unsigned int* indices, int vCount, int cCount, FileOutputStream& O, BVH_Construction_Result* out = 0);

}