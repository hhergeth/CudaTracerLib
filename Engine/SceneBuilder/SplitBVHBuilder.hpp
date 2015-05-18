#pragma once
#include "..\..\MathTypes.h"
#include "..\e_IntersectorData.h"
#include "../../Base/Timer.h"

class IBVHBuilderCallback
{
public:
	virtual void getBox(unsigned int index, AABB* out) const = 0;
	virtual unsigned int handleLeafObjects(unsigned int pNode)
	{
		return pNode;
	}
	virtual void handleLastLeafObject()
	{
	}
	virtual unsigned int Count() const = 0;
	virtual void HandleBoundingBox(const AABB& box)
	{
	}
	virtual e_BVHNodeData* HandleNodeAllocation(int* index) = 0;
	virtual void HandleStartNode(int startNode) = 0;
	virtual bool SplitNode(unsigned int index, int dim, float pos, AABB& lBox, AABB& rBox, const AABB& refBox) const
	{
		return false;
	}
};

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
	public:
		BVHNode() : m_probability(1.f), m_parentProbability(1.f), m_treelet(-1), m_index(-1) {}
		virtual bool        isLeaf() const = 0;
		virtual int         getNumChildNodes() const = 0;
		virtual BVHNode*    getChildNode(int i) const = 0;
		virtual int         getNumTriangles() const { return 0; }

		float       getArea() const     { return m_bounds.Area(); }

		AABB        m_bounds;

		// These are somewhat experimental, for some specific test and may be invalid...
		float       m_probability;          // probability of coming here (widebvh uses this)
		float       m_parentProbability;    // probability of coming to parent (widebvh uses this)

		int         m_treelet;              // for queuing tests (qmachine uses this)
		int         m_index;                // in linearized tree (qmachine uses this)

		// Subtree functions
		int     getSubtreeSize(BVH_STAT stat = BVH_STAT_NODE_COUNT) const;
		void    computeSubtreeProbabilities(const Platform& p, float parentProbability, float& sah);
		float   computeSubtreeSAHCost(const Platform& p) const;     // NOTE: assumes valid probabilities
		void    deleteSubtree();

		void    assignIndicesDepthFirst(int index = 0, bool includeLeafNodes = true);
		void    assignIndicesBreadthFirst(int index = 0, bool includeLeafNodes = true);
	};


	class InnerNode : public BVHNode
	{
	public:
		InnerNode(const AABB& bounds, BVHNode* child0, BVHNode* child1)   { m_bounds = bounds; m_children[0] = child0; m_children[1] = child1; }
		~InnerNode()
		{
			if (m_children[0])
				delete m_children[0];
			if (m_children[1])
				delete m_children[1];
		}

		bool        isLeaf() const                  { return false; }
		int         getNumChildNodes() const        { return 2; }
		BVHNode*    getChildNode(int i) const       { CT_ASSERT(i >= 0 && i<2); return m_children[i]; }

		BVHNode*    m_children[2];
	};


	class LeafNode : public BVHNode
	{
	public:
		LeafNode(const AABB& bounds, int lo, int hi)  { m_bounds = bounds; m_lo = lo; m_hi = hi; }
		LeafNode(const LeafNode& s)                 { *this = s; }

		bool        isLeaf() const                  { return true; }
		int         getNumChildNodes() const        { return 0; }
		BVHNode*    getChildNode(int) const         { return NULL; }

		int         getNumTriangles() const         { return m_hi - m_lo; }
		int         m_lo;
		int         m_hi;
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

	BVHNode*                buildNode(NodeSpec spec, int level, float progressStart, float progressEnd);
    BVHNode*                createLeaf          (const NodeSpec& spec);

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

	cTimer m_Timer;
};
