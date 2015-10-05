#pragma once
#include "..\..\MathTypes.h"
#include "..\e_TriIntersectorData.h"
#include "../../Base/Timer.h"
#include <vector>
#include <functional>

class IBVHBuilderCallback
{
public:
	virtual void getBox(unsigned int index, AABB* out) const = 0;
	virtual void iterateObjects(std::function<void(unsigned int)> f) = 0;
	virtual unsigned int handleLeafObjects(unsigned int pNode)
	{
		return pNode;
	}
	virtual void handleLastLeafObject(int parent)
	{
	}
	virtual void HandleBoundingBox(const AABB& box)
	{
	}
	virtual e_BVHNodeData* HandleNodeAllocation(int* index) = 0;
	virtual void HandleStartNode(int startNode) = 0;
	virtual bool SplitNode(unsigned int index, int dim, float pos, AABB& lBox, AABB& rBox, const AABB& refBox) const
	{
		return false;
	}
	virtual void setSibling(int idx, int sibling) = 0;
	virtual void setNumInner_Leaf(unsigned int nInnerNodes, unsigned int nLeafNodes)
	{

	}
};

/*
Copyright (c) 2009-2011, NVIDIA Corporation
All rights reserved.
*/
//These classes are based on "understanding-the-efficiency-of-ray-traversal-on-gpus" with slight modifications.
//The main algorithm is 100% their great work!

enum BVH_STAT
{
	BVH_STAT_NODE_COUNT,
	BVH_STAT_INNER_COUNT,
	BVH_STAT_LEAF_COUNT,
	BVH_STAT_TRIANGLE_COUNT,
	BVH_STAT_CHILDNODE_COUNT,
};

class BVHNode
{
	unsigned int left, right;
public:
	AABB box;
	BVHNode(const AABB& bounds, unsigned int l, unsigned int r, bool leaf) : box(bounds), left((l << 1) | unsigned int(leaf)), right(r) {}
	bool isLeaf() { return left & 1; }
	unsigned int getLeft(){ return left >> 1; }
	unsigned int getRight(){ return right; }
};

class SplitBVHBuilder
{
private:
	enum
	{
		MaxDepth        = 64,
		MaxSpatialDepth = 48,
		NumSpatialBins  = 128,
	};

	struct Reference
	{
		int                 triIdx;
		AABB                bounds;

		Reference(void) : triIdx(-1) { bounds = AABB::Identity(); }
	};

	struct NodeSpec
	{
		int                 numRef;
		AABB                bounds;

		NodeSpec(void) : numRef(0) { bounds = AABB::Identity(); }
	};

	struct ObjectSplit
	{
		float                 sah;
		int                 sortDim;
		int                 numLeft;
		AABB                leftBounds;
		AABB                rightBounds;

		ObjectSplit(void) : sah(FLT_MAX), sortDim(0), numLeft(0) { leftBounds = AABB::Identity(); rightBounds = AABB::Identity(); }
	};

	struct SpatialSplit
	{
		float                 sah;
		int                 dim;
		float                 pos;

		SpatialSplit(void) : sah(FLT_MAX), dim(0), pos(0.0f) {}
	};

	struct SpatialBin
	{
		AABB                bounds;
		int                 enter;
		int                 exit;
	};

public:

	class Platform
	{
	public:
		Platform()                                                                                                          { m_SAHNodeCost = 1.f; m_SAHTriangleCost = 1.f; m_nodeBatchSize = 1; m_triBatchSize = 1; m_minLeafSize = 1; m_maxLeafSize = 0x7FFFFFF; }
		//Platform(float nodeCost=1.f, float triCost=1.f, int nodeBatchSize=1, int triBatchSize=1) { m_SAHNodeCost = nodeCost; m_SAHTriangleCost = triCost; m_nodeBatchSize = nodeBatchSize; m_triBatchSize = triBatchSize; m_minLeafSize=1; m_maxLeafSize=0x7FFFFFF; }


		// SAH weights
		float getSAHTriangleCost() const                    { return m_SAHTriangleCost; }
		float getSAHNodeCost() const                        { return m_SAHNodeCost; }

		// SAH costs, raw and batched
		float getCost(int numChildNodes, int numTris) const  { return getNodeCost(numChildNodes) + getTriangleCost(numTris); }
		float getTriangleCost(int n) const                  { return roundToTriangleBatchSize(n) * m_SAHTriangleCost; }
		float getNodeCost(int n) const                      { return roundToNodeBatchSize(n) * m_SAHNodeCost; }

		// batch processing (how many ops at the price of one)
		int   getTriangleBatchSize() const                  { return m_triBatchSize; }
		int   getNodeBatchSize() const                      { return m_nodeBatchSize; }
		void  setTriangleBatchSize(int triBatchSize)        { m_triBatchSize = triBatchSize; }
		void  setNodeBatchSize(int nodeBatchSize)           { m_nodeBatchSize = nodeBatchSize; }
		int   roundToTriangleBatchSize(int n) const         { return ((n + m_triBatchSize - 1) / m_triBatchSize)*m_triBatchSize; }
		int   roundToNodeBatchSize(int n) const             { return ((n + m_nodeBatchSize - 1) / m_nodeBatchSize)*m_nodeBatchSize; }

		// leaf preferences
		void  setLeafPreferences(int minSize, int maxSize)   { m_minLeafSize = minSize; m_maxLeafSize = maxSize; }
		int   getMinLeafSize() const                        { return m_minLeafSize; }
		int   getMaxLeafSize() const                        { return m_maxLeafSize; }

	public:
		float   m_SAHNodeCost;
		float   m_SAHTriangleCost;
		int     m_triBatchSize;
		int     m_nodeBatchSize;
		int     m_minLeafSize;
		int     m_maxLeafSize;
	};
	struct Stats
	{
		Stats()             { clear(); }
		void clear()        { memset(this, 0, sizeof(Stats)); }
		void print() const  { printf("Tree stats: [bfactor=%d] %d nodes (%d+%d), %.2f SAHCost, %.1f children/inner, %.1f tris/leaf\n", branchingFactor, numLeafNodes + numInnerNodes, numLeafNodes, numInnerNodes, SAHCost, 1.f*numChildNodes / max(numInnerNodes, 1), 1.f*numTris / max(numLeafNodes, 1)); }

		float   SAHCost;
		int     branchingFactor;
		int     numInnerNodes;
		int     numLeafNodes;
		int     numChildNodes;
		int     numTris;
	};
	struct BuildParams
	{
		Stats*      stats;
		bool        enablePrints;
		float       splitAlpha;     // spatial split area threshold

		BuildParams(void)
		{
			stats = NULL;
			enablePrints = true;
			splitAlpha = 1.0e-5f;
		}
	};

	SplitBVHBuilder(IBVHBuilderCallback* clb, const Platform& P, const BuildParams& stats);
							~SplitBVHBuilder    (void);

	void                run                 (void);

private:
	static bool             sortCompare         (void* data, int idxA, int idxB);
	static void             sortSwap            (void* data, int idxA, int idxB);

	unsigned int                buildNode(NodeSpec spec, int level, float progressStart, float progressEnd);
	unsigned int                createLeaf(const NodeSpec& spec);

	ObjectSplit             findObjectSplit(const NodeSpec& spec, float nodeSAH);
	void                    performObjectSplit  (NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const ObjectSplit& split);

	SpatialSplit            findSpatialSplit(const NodeSpec& spec, float nodeSAH);
	void                    performSpatialSplit (NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const SpatialSplit& split);
	void                    splitReference(Reference& left, Reference& right, const Reference& ref, int dim, float pos);

private:
							SplitBVHBuilder     (const SplitBVHBuilder&); // forbidden
	SplitBVHBuilder&        operator=           (const SplitBVHBuilder&); // forbidden

private:
	IBVHBuilderCallback*	m_pClb;
	const Platform& m_platform;
	BuildParams	m_params;

	std::vector<Reference>  m_refStack;
	float                   m_minOverlap;
	std::vector<AABB>       m_rightBounds;
	int                     m_sortDim;
	SpatialBin              m_bins[3][NumSpatialBins];

	int                     m_numDuplicates;

	std::vector<int> m_Indices;

	std::vector<BVHNode> m_Nodes;

	InstructionTimer m_Timer;
};
