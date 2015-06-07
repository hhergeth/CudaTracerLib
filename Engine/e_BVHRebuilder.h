#pragma once

#include <MathTypes.h>
#include <set>

class ISpatialInfoProvider
{
public:
	AABB getBox(unsigned int idx);
	unsigned int getCount();
};

struct e_BVHNodeData;

class e_BVHRebuilder
{
	struct BVHIndex;
	struct BVHIndexTuple;

	e_BVHNodeData* m_pBVHData;
	unsigned int m_uBVHDataLength;
	unsigned int m_uBvhNodeCount;

	ISpatialInfoProvider* m_pData;
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

public:
	e_BVHRebuilder(e_BVHNodeData* data, unsigned int a_BVHNodeLength, unsigned int a_SceneNodeLength);
	~e_BVHRebuilder();

	bool Build(ISpatialInfoProvider* data);
	int getStartNode(){ return startNode; }
	AABB getBox();
	bool needsBuild();

	void addNode(unsigned int n);
	void removeNode(unsigned int n);
	void invalidateNode(unsigned int n);
private:
	void removeNodeAndCollapse(BVHIndex nodeIdx, BVHIndex childIdx);
	void insertNode(BVHIndex bvhNodeIdx, unsigned int nodeIdx, const AABB& nodeWorldBox);
	void recomputeNode(BVHIndex bvhNodeIdx, AABB& newBox);
	int getChildIdxInLocal(BVHIndex nodeIdx, BVHIndex childIdx);
	void setChild(BVHIndex nodeIdx, BVHIndex childIdx, int localIdxToSetTo);
	void sahModified(BVHIndex nodeIdx, const AABB& box, float& leftSAH, float& rightSAH);
	int validateTree(BVHIndex idx, BVHIndex parent);
	void propagateBBChange(BVHIndex idx, const AABB& box, int localChildIdx);
	AABB getBox(BVHIndex idx);
	void propagateFlag(BVHIndex idx);
	int numberGrandchildren(BVHIndex idx, int localChildIdx);
	float sah(BVHIndex lhs, BVHIndex rhs);
	float sah(BVHIndex lhs, int localChildIdx, int localGrandChildIdx);
	BVHIndexTuple children(BVHIndex idx);
	void swapChildren(BVHIndex idx, int localChildIdx, int localGrandChildIdx);
	int numLeafs(BVHIndex idx);
};