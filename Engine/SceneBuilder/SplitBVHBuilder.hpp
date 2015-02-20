#pragma once
#include "..\..\MathTypes.h"
#include "..\..\Base\BVHBuilder.h"

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

        Reference(void) : triIdx(-1) {}
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
	SplitBVHBuilder(IBVHBuilderCallback* clb, const BVHBuilder::Platform& P, const BVHBuilder::BuildParams& stats);
                            ~SplitBVHBuilder    (void);

    BVHNode*                run                 (void);

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
	const BVHBuilder::Platform& m_platform;
	BVHBuilder::BuildParams	m_params;

    std::vector<Reference>  m_refStack;
	float                   m_minOverlap;
	std::vector<AABB>       m_rightBounds;
	int                     m_sortDim;
    SpatialBin              m_bins[3][NumSpatialBins];

    int                     m_numDuplicates;

	std::vector<int> m_Indices;
};
