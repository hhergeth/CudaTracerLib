#pragma once

#include <Math/AABB.h>
#include <set>
#include <functional>
#include <fstream>
#include <bitset>

namespace CudaTracerLib {

class ISpatialInfoProvider
{
public:
	virtual AABB getBox(unsigned int idx) = 0;
	virtual void iterateObjects(std::function<void(unsigned int)> f) = 0;
	virtual void setObject(unsigned int a_IntersectorIdx, unsigned int a_ObjIdx)
	{

	}
	virtual bool SplitNode(unsigned int a_ObjIdx, int dim, float pos, AABB& lBox, AABB& rBox, const AABB& refBox) const
	{
		return false;
	}
};

struct BVHNodeData;
struct TriIntersectorData2;
class Mesh;

class BVHRebuilder
{
	struct BVHIndex;
	struct BVHIndexTuple;

	BVHNodeData* m_pBVHData;
	unsigned int m_uBVHDataLength;
	unsigned int m_uBvhNodeCount;

	TriIntersectorData2* m_pBVHIndices;
	unsigned int m_uBVHIndicesLength;
	unsigned int m_UBVHIndicesCount;

	ISpatialInfoProvider* m_pData;
	int startNode;

	struct BVHNodeInfo;
	class BuilderCLB;
	struct SceneInfo;

	std::vector<BVHNodeInfo> bvhNodeData;
	std::vector<std::vector<BVHIndex>> objectToBVHNodes;

	enum{ MAX_NODES = 1024 * 1024 * 32 };

	std::bitset<MAX_NODES> nodesToRecompute;
	std::set<unsigned int> nodesToInsert;
	std::set<unsigned int> nodesToRemove;

	std::bitset<2 * MAX_NODES> flaggedBVHNodes;
	unsigned int m_uModifiedCount;
	bool recomputeAll;

public:
	CTL_EXPORT BVHRebuilder(BVHNodeData* data, unsigned int a_BVHNodeLength, unsigned int a_SceneNodeLength, TriIntersectorData2* indices, unsigned int a_IndicesLength);
	CTL_EXPORT BVHRebuilder(Mesh* mesh, ISpatialInfoProvider* data);
	CTL_EXPORT ~BVHRebuilder();

	CTL_EXPORT bool Build(ISpatialInfoProvider* data, bool invalidateAll = false);
	CTL_EXPORT void SetEmpty();
	CTL_EXPORT bool needsBuild() const;
	CTL_EXPORT void printGraph(const std::string& path);

	CTL_EXPORT void addNode(unsigned int n);
	CTL_EXPORT void removeNode(unsigned int n);
	CTL_EXPORT void invalidateNode(unsigned int n);

	const std::bitset<MAX_NODES>& getInvalidatedNodes() const { return nodesToRecompute; }
	unsigned int getNumBVHIndicesUsed() const { return m_UBVHIndicesCount; }
	int getStartNode() const{ return startNode; }
	unsigned int getNumBVHNodesUsed() const { return m_uBvhNodeCount; }
	CTL_EXPORT AABB getBox() const;
private:
	int BuildInfoTree(BVHIndex idx, BVHIndex parent);
	void removeNodeAndCollapse(BVHIndex nodeIdx, BVHIndex childIdx);
	void insertNode(BVHIndex bvhNodeIdx, BVHIndex parent, unsigned int nodeIdx, const AABB& nodeWorldBox);
	void recomputeNode(BVHIndex bvhNodeIdx, AABB& newBox);
	int getChildIdxInLocal(BVHIndex nodeIdx, BVHIndex childIdx);
	void setChild(BVHIndex nodeIdx, BVHIndex childIdx, int localIdxToSetTo, BVHIndex oldParent, bool prop = true);
	void sahModified(BVHIndex nodeIdx, const AABB& box, float& leftSAH, float& rightSAH);
	int validateTree(BVHIndex idx, BVHIndex parent, bool checkAABB = true);
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
	bool removeObjectFromLeaf(BVHIndex leafIdx, unsigned int objIdx);
	void writeGraphPart(BVHIndex idx, BVHIndex parent, std::ofstream& f);
};

}