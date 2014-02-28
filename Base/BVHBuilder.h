#pragma once

#include "..\MathTypes.h"

struct e_BVHNodeData
{
//      nodes[innerOfs + 0 ] = Vec4f(c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
//      nodes[innerOfs + 16] = Vec4f(c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
//      nodes[innerOfs + 32] = Vec4f(c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
	float4 a,b,c,d;
	int2 getChildren()
	{
		return *(int2*)&d;
	}
	CUDA_FUNC_IN void getBox(AABB& left, AABB& right)
	{
		left.minV = make_float3(a.x, a.z, c.x);
		left.maxV = make_float3(a.y, a.w, c.y);
		right.minV = make_float3(b.x, b.z, c.z);
		right.maxV = make_float3(b.y, b.w, c.w);
	}
	CUDA_FUNC_IN AABB getLeft()
	{
		return AABB(make_float3(a.x, a.z, c.x), make_float3(a.y, a.w, c.y));
	}
	CUDA_FUNC_IN AABB getRight()
	{
		return AABB(make_float3(b.x, b.z, c.z), make_float3(b.y, b.w, c.w));
	}
	CUDA_FUNC_IN void setBox(AABB& c0, AABB& c1)
	{
		a = make_float4(c0.minV.x, c0.maxV.x, c0.minV.y, c0.maxV.y);
		b = make_float4(c1.minV.x, c1.maxV.x, c1.minV.y, c1.maxV.y);
		c = make_float4(c0.minV.z, c0.maxV.z, c1.minV.z, c1.maxV.z);
	}
	CUDA_FUNC_IN void setLeft(AABB& c0)
	{
		a = make_float4(c0.minV.x, c0.maxV.x, c0.minV.y, c0.maxV.y);
		c.x = c0.minV.z;
		c.y = c0.maxV.z;
	}
	CUDA_FUNC_IN void setRight(AABB& c1)
	{
		b = make_float4(c1.minV.x, c1.maxV.x, c1.minV.y, c1.maxV.y);
		c.z = c1.minV.z;
		c.w = c1.maxV.z;
	}
	CUDA_FUNC_IN void setChildren(int2 c)
	{
		*(int2*)&d = c;
	}
	void setDummy()
	{
		AABB std(make_float3(0), make_float3(0));
		setLeft(std);
		setRight(std);
		setChildren(make_int2(0,0));
	}
};

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
	virtual bool SplitNode(unsigned int index, int dim, float pos, AABB& lBox, AABB& rBox) const
	{
		return false;
	}
};

class BVHBuilder
{
public:
	class Platform
	{
	public:
		Platform()                                                                                                          { m_SAHNodeCost = 1.f; m_SAHTriangleCost = 1.f; m_nodeBatchSize = 1; m_triBatchSize = 1; m_minLeafSize=1; m_maxLeafSize=0x7FFFFFF; }
		//Platform(float nodeCost=1.f, float triCost=1.f, int nodeBatchSize=1, int triBatchSize=1) { m_SAHNodeCost = nodeCost; m_SAHTriangleCost = triCost; m_nodeBatchSize = nodeBatchSize; m_triBatchSize = triBatchSize; m_minLeafSize=1; m_maxLeafSize=0x7FFFFFF; }


		// SAH weights
		float getSAHTriangleCost() const                    { return m_SAHTriangleCost; }
		float getSAHNodeCost() const                        { return m_SAHNodeCost; }

		// SAH costs, raw and batched
		float getCost(int numChildNodes,int numTris) const  { return getNodeCost(numChildNodes) + getTriangleCost(numTris); }
		float getTriangleCost(int n) const                  { return roundToTriangleBatchSize(n) * m_SAHTriangleCost; }
		float getNodeCost(int n) const                      { return roundToNodeBatchSize(n) * m_SAHNodeCost; }

		// batch processing (how many ops at the price of one)
		int   getTriangleBatchSize() const                  { return m_triBatchSize; }
		int   getNodeBatchSize() const                      { return m_nodeBatchSize; }
		void  setTriangleBatchSize(int triBatchSize)        { m_triBatchSize = triBatchSize; }
		void  setNodeBatchSize(int nodeBatchSize)           { m_nodeBatchSize= nodeBatchSize; }
		int   roundToTriangleBatchSize(int n) const         { return ((n+m_triBatchSize-1)/m_triBatchSize)*m_triBatchSize; }
		int   roundToNodeBatchSize(int n) const             { return ((n+m_nodeBatchSize-1)/m_nodeBatchSize)*m_nodeBatchSize; }

		// leaf preferences
		void  setLeafPreferences(int minSize,int maxSize)   { m_minLeafSize=minSize; m_maxLeafSize=maxSize; }
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
	static void BuildBVH(IBVHBuilderCallback* clb, const BVHBuilder::Platform& P);
};