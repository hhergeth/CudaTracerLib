#pragma once
#include <Math/AABB.h>
#include <Engine/TriIntersectorData.h>
#include <Base/Timer.h>
#include <vector>
#include <functional>

namespace CudaTracerLib {

class IBVHBuilderCallback
{
public:

	virtual ~IBVHBuilderCallback()
	{

	}

	///reports the number of inner/leaf nodes to the clb to allocate enough storage
	virtual void startConstruction(unsigned int nInnerNodes, unsigned int nLeafNodes) = 0;

	///iterate over all objects and call f with the idx and appropriate aabb
	virtual void iterateObjects(std::function<void(unsigned int, const AABB&)> f) = 0;

	///create a leaf node unter the specified inner node \ref parentBVHNodeIdx using the object indices \ref objIndices, returning the first index in the object index buffer
	virtual unsigned int createLeafNode(unsigned int parentBVHNodeIdx, const std::vector<unsigned int>& objIndices) = 0;

	///reports the end of bvh construction and provides the idx of root node as well as the aabb containing all objects
	virtual void finishConstruction(unsigned int startNode, const AABB& sceneBox) = 0;

	///creates an inner node and returns its idx as well as a ref to it
	virtual unsigned int createInnerNode(BVHNodeData*& innerNode) = 0;

	///splits the obj with \ref objIdx if possible in the specified dimension \ref dim at \pos and returns the left and right aabb in \ref lBox respectively \rBox
	virtual bool SplitNode(unsigned int objIdx, int dim, float pos, AABB& lBox, AABB& rBox, const AABB& refBox) const
	{
		return false;
	}
};

/*
Copyright (c) 2009-2011, NVIDIA Corporation
All rights reserved.
*/
//These classes are based on "understanding-the-efficiency-of-ray-traversal-on-gpus" with slight modifications.
//The main algorithm is 100% their great work!

class BVHNode
{
	unsigned int left, right;
public:
	AABB box;
	BVHNode(const AABB& bounds, unsigned int l, unsigned int r, bool leaf) : left((l << 1) | (unsigned int)leaf), right(r), box(bounds) {}
	bool isLeaf() const { return left & 1; }
	unsigned int getLeft() const { return left >> 1; }
	unsigned int getRight() const { return right; }
};

class SplitBVHBuilder
{
private:
	enum
	{
		MaxDepth = 64,
		MaxSpatialDepth = 48,
		NumSpatialBins = 128,
	};

	struct Reference
	{
		int                 triIdx;
		AABB                bounds;

		Reference(void) : triIdx(-1), bounds(AABB::Identity()) { }
	};

	struct NodeSpec
	{
		int                 numRef;
		AABB                bounds;

		NodeSpec(void) : numRef(0), bounds(AABB::Identity()) { }
	};

	struct ObjectSplit
	{
		float                 sah;
		int                 sortDim;
		int                 numLeft;
		AABB                leftBounds;
		AABB                rightBounds;

		ObjectSplit(void) : sah(FLT_MAX), sortDim(0), numLeft(0), leftBounds(AABB::Identity()), rightBounds(AABB::Identity()) { }
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
		Platform()                                                                                                          { m_SAHNodeCost = 1.f; m_SAHTriangleCost = 1.f; m_nodeBatchSize = 1; m_triBatchSize = 1; m_minLeafSize = 1; m_maxLeafSize = 0x7FFFFFF; m_objectSplits = m_spatialSplits = true; }
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

		bool	m_objectSplits;
		bool	m_spatialSplits;
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
		void clear()        { CudaTracerLib::Platform::SetMemory(this, sizeof(Stats)); }
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

	CTL_EXPORT SplitBVHBuilder(IBVHBuilderCallback* clb, const Platform& P, const BuildParams& stats);
	~SplitBVHBuilder(void);

	CTL_EXPORT void                run(void);

private:
	static bool             sortCompare(void* data, int idxA, int idxB);
	static void             sortSwap(void* data, int idxA, int idxB);

	unsigned int                buildNode(NodeSpec spec, int level, float progressStart, float progressEnd);
	unsigned int                createLeaf(const NodeSpec& spec);

	ObjectSplit             findObjectSplit(const NodeSpec& spec, float nodeSAH);
	void                    performObjectSplit(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const ObjectSplit& split);

	SpatialSplit            findSpatialSplit(const NodeSpec& spec, float nodeSAH);
	void                    performSpatialSplit(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const SpatialSplit& split);
	void                    splitReference(Reference& left, Reference& right, const Reference& ref, int dim, float pos);

private:
	SplitBVHBuilder(const SplitBVHBuilder&); // forbidden
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

}
