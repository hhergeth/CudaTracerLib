#pragma once

#include <MathTypes.h>
#include <set>
#include <functional>
#include <fstream>
#include <bitset>

class ISpatialInfoProvider
{
public:
	virtual AABB getBox(unsigned int idx) = 0;
	virtual unsigned int getCount() = 0;
	virtual void setObject(unsigned int a_IntersectorIdx, unsigned int a_ObjIdx)
	{

	}
	virtual bool SplitNode(unsigned int a_ObjIdx, int dim, float pos, AABB& lBox, AABB& rBox, const AABB& refBox) const
	{
		return false;
	}
};

struct e_BVHNodeData;
struct e_TriIntersectorData2;

class e_BVHRebuilder
{
	struct BVHIndex;
	struct BVHIndexTuple;

	e_BVHNodeData* m_pBVHData;
	unsigned int m_uBVHDataLength;
	unsigned int m_uBvhNodeCount;

	e_TriIntersectorData2* m_pBVHIndices;
	unsigned int m_uBVHIndicesLength;
	unsigned int m_UBVHIndicesCount;

	ISpatialInfoProvider* m_pData;
	int startNode;

	struct BVHNodeInfo;
	class BuilderCLB;
	struct SceneInfo;

	std::vector<BVHNodeInfo> bvhNodeData;
	std::vector<BVHIndex> nodeToBVHNode;

	enum{MAX_NODES = 1024 * 1024 * 32};

	std::bitset<MAX_NODES> nodesToRecompute;
	std::set<unsigned int> nodesToInsert;
	std::set<unsigned int> nodesToRemove;

	std::bitset<2 * MAX_NODES> flaggedBVHNodes;
	unsigned int m_uModifiedCount;
	bool recomputeAll;

public:
	e_BVHRebuilder(e_BVHNodeData* data, unsigned int a_BVHNodeLength, unsigned int a_SceneNodeLength, e_TriIntersectorData2* indices, unsigned int a_IndicesLength);
	~e_BVHRebuilder();

	bool Build(ISpatialInfoProvider* data);
	int getStartNode(){ return startNode; }
	unsigned int getNumBVHNodesUsed(){ return m_uBvhNodeCount; }
	AABB getBox();
	bool needsBuild();
	void printGraph(const std::string& path);

	void addNode(unsigned int n);
	void removeNode(unsigned int n);
	void invalidateNode(unsigned int n);

	const std::bitset<MAX_NODES>& getInvalidatedNodes(){ return nodesToRecompute; }
private:
	void removeNodeAndCollapse(BVHIndex nodeIdx, BVHIndex childIdx);
	void insertNode(BVHIndex bvhNodeIdx, BVHIndex parent, unsigned int nodeIdx, const AABB& nodeWorldBox);
	void recomputeNode(BVHIndex bvhNodeIdx, AABB& newBox);
	int getChildIdxInLocal(BVHIndex nodeIdx, BVHIndex childIdx);
	void setChild(BVHIndex nodeIdx, BVHIndex childIdx, int localIdxToSetTo);
	void sahModified(BVHIndex nodeIdx, const AABB& box, float& leftSAH, float& rightSAH);
	int validateTree(BVHIndex idx, BVHIndex parent);
	void propagateBBChange(BVHIndex idx, const AABB& box, int localChildIdx);
	AABB getBox(BVHIndex idx);
	void propagateFlag(BVHIndex idx);
	int numberGrandchildren(BVHIndex idx, int localChildIdx);
	float sah(BVHIndex lhs, int localChildIdx, int localGrandChildIdx);
	BVHIndexTuple children(BVHIndex idx);
	void swapChildren(BVHIndex idx, int localChildIdx, int localGrandChildIdx);
	int numLeafs(BVHIndex idx);
	void enumerateLeaf(BVHIndex idx, const std::function<void(int)>& f);
	BVHIndex createLeaf(unsigned int nodeIdx, BVHIndex oldLeaf);
	BVHIndex getLeafIdx(unsigned int objIdx);
	bool removeObjectFromLeaf(BVHIndex leafIdx, unsigned int objIdx);
	void writeGraphPart(BVHIndex idx, BVHIndex parent, std::ofstream& f);
};