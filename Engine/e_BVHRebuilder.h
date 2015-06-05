#pragma once

#include <MathTypes.h>
#include <set>

class IBVHBuilderCallback;

class e_BVHRebuilder
{
	struct BVHIndex;
	struct BVHIndexTuple;

	int m_sBvhNodeCount;
	IBVHBuilderCallback* m_pData;
	int startNode;

	struct BVHNodeInfo;
	class BuilderCLB;
	struct SceneInfo;

	std::vector<BVHNodeInfo> bvhNodeData;
	std::vector<BVHIndex> nodeToBVHNode;

	std::set<unsigned int> nodesToRecompute;
	std::set<unsigned int> nodesToInsert;
	std::set<unsigned int> nodesToRemove;
	std::vector<bool> flaggedBVHNodes, flaggedSceneNodes;
};