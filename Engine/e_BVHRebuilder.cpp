#include <StdAfx.h>
#include "e_BVHRebuilder.h"
#include "SceneBuilder/SplitBVHBuilder.hpp"
#include <algorithm>

#define NO_NODE 0x76543210

//bvh tree rotations from
//http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.331.9382&rep=rep1&type=pdf

struct e_BVHRebuilder::BVHIndex
{
private:
	int native;
	BVHIndex(int i)
		: native(i)
	{

	}
public:
	BVHIndex(){}
	static BVHIndex FromNative(int i)
	{
		return BVHIndex(i);
	}
	static BVHIndex FromSceneNode(unsigned int n)
	{
		return BVHIndex(~n);
	}
	static BVHIndex FromBVHNode(unsigned int n)
	{
		return BVHIndex(n * 4);
	}
	bool isLeaf() const
	{
		return native < 0;
	}
	bool NoNode() const
	{
		return native == NO_NODE;
	}
	int ToNative() const
	{
		return native;
	}
	int innerIdx() const
	{
		if (isLeaf())
			throw std::runtime_error("not an inner node!");
		return native / 4;
	}
	int leafIdx() const
	{
		if (!isLeaf())
			throw std::runtime_error("not a leaf!");
		return ~native;
	}
	bool isValid() const
	{
		return native != -1;
	}
	bool operator==(const BVHIndex& rhs) const
	{
		return native == rhs.native;
	}
	bool operator!=(const BVHIndex& rhs) const
	{
		return native != rhs.native;
	}
	static BVHIndex INVALID()
	{
		return FromNative(-1);
	}
};

struct e_BVHRebuilder::BVHIndexTuple
{
	BVHIndex values[2];

	BVHIndex& operator[](int i)
	{
		return values[i];
	}

	const BVHIndex& operator[](int i) const
	{
		return values[i];
	}

	static BVHIndexTuple FromChildren(const e_BVHNodeData* n)
	{
		BVHIndexTuple t;
		for (int i = 0; i < 2; i++)
			t.values[i] = BVHIndex::FromNative(n->getChildren()[i]);
		return t;
	}
};

struct e_BVHRebuilder::BVHNodeInfo
{
	int numLeafs;

	BVHNodeInfo()
		: numLeafs(0)
	{

	}

	BVHNodeInfo(int l)
		: numLeafs(l)
	{

	}

	static void changeCount(std::vector<BVHNodeInfo>& nodeInfos, const e_BVHNodeData* nodeData, BVHIndex idx, int off)
	{
		nodeInfos[idx.innerIdx()].numLeafs += off;
		BVHIndex parent = BVHIndex::FromNative(nodeData[idx.innerIdx()].getParent());
		if (parent.isValid())
			changeCount(nodeInfos, nodeData, parent, off);
	}
};

class e_BVHRebuilder::BuilderCLB : public IBVHBuilderCallback
{
	e_BVHRebuilder* t;
public:
	BuilderCLB(e_BVHRebuilder* c)
		: t(c)
	{
		t->m_uBvhNodeCount = 0;
		t->m_UBVHIndicesCount = 0;
	}
	virtual unsigned int Count() const
	{
		return t->m_pData->getCount();
	}
	virtual void getBox(unsigned int index, AABB* out) const
	{
		*out = t->m_pData->getBox(index);
	}
	virtual void HandleBoundingBox(const AABB& box)
	{
		
	}
	virtual e_BVHNodeData* HandleNodeAllocation(int* index)
	{
		if (t->m_uBvhNodeCount >= t->m_uBVHDataLength)
			throw std::runtime_error(__FUNCTION__);
		*index = t->m_uBvhNodeCount * 4;
		return t->m_pBVHData + t->m_uBvhNodeCount++;
	}
	virtual void setSibling(int idx, int sibling)
	{
		if (idx >= 0)
			t->m_pBVHData[idx / 4].setSibling(sibling);
	}
	virtual unsigned int handleLeafObjects(unsigned int pNode)
	{
		if (t->m_pBVHIndices)
		{
			unsigned int c = t->m_UBVHIndicesCount++;
			t->m_pBVHIndices[c].setFlag(false);
			t->m_pBVHIndices[c].setIndex(pNode);
			t->m_pData->setObject(c, pNode);
			return c;
		}
		else return pNode;

		return pNode;
	}
	virtual void handleLastLeafObject(int parent)
	{
		if (t->m_pBVHIndices)
			t->m_pBVHIndices[t->m_UBVHIndicesCount - 1].setFlag(true);
	}
	virtual void HandleStartNode(int startNode)
	{
		t->startNode = startNode;
	}
	virtual bool SplitNode(unsigned int a_ObjIdx, int dim, float pos, AABB& lBox, AABB& rBox, const AABB& refBox) const
	{
		return t->m_pData->SplitNode(a_ObjIdx, dim, pos, lBox, rBox, refBox);
	}
	int buildInfoTree(BVHIndex idx, BVHIndex parent, e_BVHRebuilder* t)
	{
		BVHIndexTuple c = BVHIndexTuple::FromChildren(t->m_pBVHData + idx.innerIdx());
		int s = 0;
		for (int i = 0; i < 2; i++)
		{
			if (c[i].isLeaf())
			{
				t->enumerateLeaf(c[i], [&](int i){t->nodeToBVHNode[i] = idx; s++; });		
			}
			else if (!c[i].NoNode())
				s += buildInfoTree(c[i], idx, t);
		}
		t->bvhNodeData[idx.innerIdx()].numLeafs = s;
		return s;
	}
};

void e_BVHRebuilder::removeNodeAndCollapse(BVHIndex nodeIdx, BVHIndex childIdx)
{
	int childIdxLocal = getChildIdxInLocal(nodeIdx, childIdx);
	e_BVHNodeData* node = m_pBVHData + nodeIdx.innerIdx();
	BVHIndexTuple children = this->children(nodeIdx);
	node->setChild(childIdxLocal, NO_NODE, AABB::Identity());
	propagateBBChange(nodeIdx, AABB::Identity(), childIdxLocal);
	BVHNodeInfo::changeCount(bvhNodeData, m_pBVHData, nodeIdx, -1);
	BVHIndex grandpaIdx = BVHIndex::FromNative(m_pBVHData[nodeIdx.innerIdx()].getParent());
	if (children[1 - childIdxLocal].NoNode() && grandpaIdx.isValid())
	{
		//remove this node and propagate?
		removeNodeAndCollapse(grandpaIdx, nodeIdx);
	}
	else
	{
		BVHIndex otherChild = children[1 - childIdxLocal];
		if (!grandpaIdx.isValid())
		{
			if (!otherChild.isLeaf() && !otherChild.NoNode())//there has to be one inner node in the tree
			{
				startNode = otherChild.ToNative();
				m_pBVHData[otherChild.innerIdx()].setParent(-1);
				m_pBVHData[otherChild.innerIdx()].setSibling(-1);
			}
		}
		else if (grandpaIdx.isValid())
		{
			int parentLocal = getChildIdxInLocal(grandpaIdx, nodeIdx);
			setChild(grandpaIdx, otherChild, parentLocal);
			propagateBBChange(grandpaIdx, getBox(nodeIdx), parentLocal);
		}
	}
}

void e_BVHRebuilder::insertNode(BVHIndex bvhNodeIdx, BVHIndex parent, unsigned int nodeIdx, const AABB& nodeWorldBox)
{
	if (bvhNodeIdx.isLeaf())
	{
		if (m_pBVHIndices)//add to leaf
		{
			//TODO sah for split?
			BVHNodeInfo::changeCount(bvhNodeData, m_pBVHData, parent, +1);
			BVHIndex newLeafIdx = createLeaf(nodeIdx, bvhNodeIdx);
			setChild(parent, newLeafIdx, getChildIdxInLocal(parent, bvhNodeIdx));
		}
		else//split leaf
		{
			e_BVHNodeData* node = m_pBVHData + m_uBvhNodeCount++;
			BVHIndex parentIdx = nodeToBVHNode[bvhNodeIdx.leafIdx()];
			int localIdx = getChildIdxInLocal(parentIdx, bvhNodeIdx);
			BVHIndex idx = BVHIndex::FromBVHNode(node - m_pBVHData);
			setChild(idx, BVHIndex::FromSceneNode(nodeIdx), 0);
			setChild(idx, bvhNodeIdx, 1);
			bvhNodeData[idx.innerIdx()] = BVHNodeInfo(1);//only set one so we can increment
			setChild(parentIdx, idx, localIdx);
			BVHNodeInfo::changeCount(bvhNodeData, m_pBVHData, idx, +1);
			nodeToBVHNode[nodeIdx] = idx;
			nodeToBVHNode[bvhNodeIdx.leafIdx()] = idx;
			propagateBBChange(parentIdx, node->getBox(), localIdx);
		}
	}
	else
	{
		e_BVHNodeData* node = m_pBVHData + bvhNodeIdx.innerIdx();
		BVHIndexTuple c = children(bvhNodeIdx);
		if (c[0].NoNode() || c[1].NoNode())//insert into one child
		{
			node->setChild(c[1].NoNode(), createLeaf(nodeIdx, BVHIndex::INVALID()).ToNative(), m_pData->getBox(nodeIdx));
			BVHNodeInfo::changeCount(bvhNodeData, m_pBVHData, bvhNodeIdx, +1);
			nodeToBVHNode[nodeIdx] = bvhNodeIdx;
			propagateBBChange(bvhNodeIdx, nodeWorldBox, c[1].NoNode());
		}
		else
		{
			float left_sah, right_sah;
			sahModified(bvhNodeIdx, nodeWorldBox, left_sah, right_sah);
			insertNode(left_sah < right_sah ? c[0] : c[1], bvhNodeIdx, nodeIdx, nodeWorldBox);
		}
	}
}

void e_BVHRebuilder::recomputeNode(BVHIndex bvhNodeIdx, AABB& newBox)
{
	if (bvhNodeIdx.isLeaf())
	{
		int c = bvhNodeIdx.leafIdx();
		enumerateLeaf(bvhNodeIdx, [&](int i){m_pData->setObject(c++, i); });
		newBox = getBox(bvhNodeIdx);
	}
	else
	{
		BVHNodeInfo& info = bvhNodeData[bvhNodeIdx.innerIdx()];
		e_BVHNodeData* node = m_pBVHData + bvhNodeIdx.innerIdx();
		BVHIndexTuple c = children(bvhNodeIdx);
		bool modified = false;
		for (int i = 0; i < 2; i++)
		{
			if (c[i].isLeaf() || (!c[i].NoNode() && (recomputeAll || flaggedBVHNodes[c[i].innerIdx()])))
			{
				modified = true;
				recomputeNode(c[i], newBox);
				if (i == 0)
					node->setLeft(newBox);
				else node->setRight(newBox);
			}
		}

		if (modified)
		{
			bool canRotateAB = numberGrandchildren(bvhNodeIdx, 0) == 2;
			bool canRotateCD = numberGrandchildren(bvhNodeIdx, 1) == 2;
			float sah_rots[] = { FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX };
			if (canRotateAB)
			{
				sah_rots[0] = sah(bvhNodeIdx, 1, 0);
				sah_rots[1] = sah(bvhNodeIdx, 1, 1);
			}
			if (canRotateCD)
			{
				sah_rots[2] = sah(bvhNodeIdx, 0, 1);
				sah_rots[3] = sah(bvhNodeIdx, 0, 0);
			}
			int best_rot = std::min_element(sah_rots, sah_rots + 4) - sah_rots;
			AABB lBox = getBox(c[0]), rBox = getBox(c[1]);
			float sah = lBox.Area() * numLeafs(c[0]) + rBox.Area() * numLeafs(c[1]);
			if (sah_rots[best_rot] < sah)
			{
				if (best_rot == 0)
					swapChildren(bvhNodeIdx, 1, 0);
				else if (best_rot == 1)
					swapChildren(bvhNodeIdx, 1, 1);
				else if (best_rot == 2)
					swapChildren(bvhNodeIdx, 0, 1);
				else if (best_rot == 3)
					swapChildren(bvhNodeIdx, 0, 0);
			}
		}
		else throw std::runtime_error("Invalid flaggednodes!");

		newBox = getBox(bvhNodeIdx);
	}
}

void e_BVHRebuilder::propagateFlag(BVHIndex idx)
{
	if (idx.isLeaf())
	{
		int objIdx = m_pBVHIndices ? m_pBVHIndices[idx.leafIdx()].getIndex() : idx.leafIdx();
		propagateFlag(nodeToBVHNode[objIdx]);
	}
	else
	{
		if (flaggedBVHNodes[idx.innerIdx()])
			return;
		flaggedBVHNodes[idx.innerIdx()] = 1;
		BVHIndex parent = BVHIndex::FromNative(m_pBVHData[idx.innerIdx()].getParent());
		if (parent.isValid())
			propagateFlag(parent);
	}
}

bool e_BVHRebuilder::Build(ISpatialInfoProvider* data)
{
	this->m_pData = data;
	bool modified = false;
	if (data->getCount() != 0 && needsBuild())
	{
		modified = true;
		size_t m = max(nodesToInsert.size(), nodesToRecompute.size(), nodesToRemove.size());
		if (startNode == -1)
		{
			BuilderCLB b(this);
			SplitBVHBuilder::Platform Pq;
			if (!m_pBVHIndices)
				Pq.m_maxLeafSize = 1;
			SplitBVHBuilder bu(&b, Pq, SplitBVHBuilder::BuildParams());
			bu.run();
			b.buildInfoTree(BVHIndex::FromNative(startNode), BVHIndex::INVALID(), this);
			validateTree(BVHIndex::FromNative(startNode), BVHIndex::INVALID());
		}
		else
		{
			typedef std::set<unsigned int>::iterator n_it;
			for (n_it it = nodesToRemove.begin(); it != nodesToRemove.end(); ++it)
			{
				BVHIndex leafIdx = getLeafIdx(*it);
				if (!m_pBVHIndices || removeObjectFromLeaf(leafIdx, *it))
					removeNodeAndCollapse(nodeToBVHNode[*it], leafIdx);
			}
			for (n_it it = nodesToInsert.begin(); it != nodesToInsert.end(); ++it)
				insertNode(BVHIndex::FromNative(startNode), BVHIndex::INVALID(), *it, data->getBox(*it));	
			recomputeAll = m_uModifiedCount > m_pData->getCount() / 2;
			if (!recomputeAll)
				for (size_t i = 0; i < m_pData->getCount(); i++)
					if (nodesToRecompute.at(i))
						propagateFlag(BVHIndex::FromSceneNode((unsigned int)i));
			AABB box;
			if (recomputeAll || flaggedBVHNodes[startNode / 4])
				recomputeNode(BVHIndex::FromNative(startNode), box);
			flaggedBVHNodes.reset();
#if DEBUG
			validateTree(BVHIndex::FromNative(startNode), BVHIndex::INVALID());
#endif
		}
		//printGraph("1.txt", a_Nodes);
	}
	if (!data->getCount())
	{
		startNode = -1;
		m_uBvhNodeCount = 0;
	}
	nodesToRecompute.reset();
	nodesToInsert.clear();
	nodesToRemove.clear();
	m_uModifiedCount = 0;
	return modified;
}

e_BVHRebuilder::e_BVHRebuilder(e_BVHNodeData* data, unsigned int a_BVHNodeLength, unsigned int a_SceneNodeLength, e_TriIntersectorData2* indices, unsigned int a_IndicesLength)
	: m_pBVHData(data), m_uBVHDataLength(a_BVHNodeLength), m_uBvhNodeCount(0), startNode(-1), m_pBVHIndices(indices), m_uBVHIndicesLength(a_IndicesLength), m_UBVHIndicesCount(0),
	  m_uModifiedCount(0)
{
	if (a_SceneNodeLength > MAX_NODES)
		throw std::runtime_error("e_BVHRebuilder too many objects!");
	nodeToBVHNode.resize(a_SceneNodeLength);
	bvhNodeData.resize(a_BVHNodeLength);
}

e_BVHRebuilder::~e_BVHRebuilder()
{

}

void e_BVHRebuilder::invalidateNode(unsigned int n)
{
	m_uModifiedCount++;
	if (nodesToInsert.find(n) == nodesToInsert.end())
		nodesToRecompute.set(n);
}

void e_BVHRebuilder::addNode(unsigned int n)
{
	nodesToInsert.insert(n);
}

void e_BVHRebuilder::removeNode(unsigned int n)
{
	nodesToRemove.insert(n);
	std::set<unsigned int>::iterator it = nodesToInsert.find(n);
	if (it != nodesToInsert.end())
		nodesToInsert.erase(it);
	nodesToRecompute.reset(n);
}

bool e_BVHRebuilder::needsBuild()
{
	return m_uModifiedCount || nodesToInsert.size() != 0 || nodesToRemove.size() != 0;
}

AABB e_BVHRebuilder::getBox()
{
	return m_pBVHData[startNode].getBox();
}

int e_BVHRebuilder::getChildIdxInLocal(BVHIndex nodeIdx, BVHIndex childIdx)
{
	BVHIndexTuple c = children(nodeIdx);
	if (c[0] != childIdx && c[1] != childIdx)
		throw std::runtime_error("Invalid tree passed!");
	return c[1] == childIdx;
}

void e_BVHRebuilder::sahModified(BVHIndex nodeIdx, const AABB& box, float& leftSAH, float& rightSAH)
{
	AABB left, right;
	e_BVHNodeData* node = m_pBVHData + nodeIdx.innerIdx();
	node->getBox(left, right);
	BVHIndexTuple c = children(nodeIdx);
	float lA = left.Area(), rA = right.Area();
	left.Enlarge(box);
	right.Enlarge(box);
	float lAd = left.Area(), rAd = right.Area();
	int lN = numLeafs(c[0]), rN = numLeafs(c[1]);
	leftSAH = lAd * (lN + 1) + rA * rN;
	rightSAH = lA * lN + rAd * (rN + 1);
}

int e_BVHRebuilder::validateTree(BVHIndex idx, BVHIndex parent)
{
	if (idx.isLeaf())
	{
		enumerateLeaf(idx, [&](int i){if (nodeToBVHNode[i] != parent) throw std::runtime_error(__FUNCTION__); });
		return numLeafs(idx);
	}
	else if (idx.NoNode())
		return 0;
	else
	{
		const BVHNodeInfo& info = bvhNodeData[idx.innerIdx()];
		BVHIndex stored_parent = BVHIndex::FromNative(m_pBVHData[idx.innerIdx()].getParent());
		if (stored_parent != parent)
			throw std::runtime_error(__FUNCTION__);
		BVHIndexTuple c = children(idx);
		int s = validateTree(c[0], idx) + validateTree(c[1], idx);
		if (s != info.numLeafs)
			throw std::runtime_error(__FUNCTION__);
		AABB leftBox = getBox(c[0]), rightBox = getBox(c[1]), box = getBox(idx), box2 = leftBox;
		box2.Enlarge(rightBox);
		if (!box.Contains(box2.minV) || !box.Contains(box2.maxV))
			throw std::runtime_error(__FUNCTION__);
		return s;
	}
}

AABB e_BVHRebuilder::getBox(BVHIndex idx)
{
	if (idx.NoNode())
		return AABB::Identity();
	if (idx.isLeaf())
	{
		AABB box = AABB::Identity();
		enumerateLeaf(idx, [&](int i){box.Enlarge(m_pData->getBox(i)); });
		return box;
	}
	else return m_pBVHData[idx.innerIdx()].getBox();
}

int e_BVHRebuilder::numberGrandchildren(BVHIndex idx, int localChildIdx)
{
	if (idx.isLeaf())
		throw std::runtime_error(__FUNCTION__);
	BVHIndexTuple c = children(idx);
	BVHIndex child = c[localChildIdx];
	if (child.isLeaf() || child.NoNode())
		return 0;
	BVHIndexTuple c2 = children(child);
	return (!c2[0].NoNode()) + (!c2[1].NoNode());
}

float e_BVHRebuilder::sah(BVHIndex idx, int localChildIdx, int localGrandChildIdx)
{
	BVHIndex childIdx = children(idx)[localChildIdx], otherChildIdx = children(idx)[1 - localChildIdx],
			 grandChildIdx = children(otherChildIdx)[localGrandChildIdx], otherGrandChildIdx = children(otherChildIdx)[1 - localGrandChildIdx];

	float childRhsArea = getBox(grandChildIdx).Area();
	int childRhsNum = numLeafs(grandChildIdx);

	AABB boxLhs1 = getBox(childIdx), boxLhs2 = getBox(otherGrandChildIdx);
	int numLhs1 = numLeafs(childIdx), numLhs2 = numLeafs(otherGrandChildIdx);
	boxLhs1.Enlarge(boxLhs2);

	return boxLhs1.Area() * (numLhs1 + numLhs2) + childRhsArea * childRhsNum;
}

e_BVHRebuilder::BVHIndexTuple e_BVHRebuilder::children(BVHIndex idx)
{
	return BVHIndexTuple::FromChildren(m_pBVHData + idx.innerIdx());
}

int e_BVHRebuilder::numLeafs(BVHIndex idx)
{
	if (idx.NoNode())
		return 0;
	if (idx.isLeaf())
	{
		int count = 0;
		enumerateLeaf(idx, [&](int i){count++; });
		return count;
	}
	else return bvhNodeData[idx.innerIdx()].numLeafs;
}

void e_BVHRebuilder::setChild(BVHIndex nodeIdx, BVHIndex childIdx, int localIdxToSetTo)
{
	m_pBVHData[nodeIdx.innerIdx()].setChild(localIdxToSetTo, childIdx.ToNative(), getBox(childIdx));
	if (childIdx.isLeaf())
		enumerateLeaf(childIdx, [&](int i){nodeToBVHNode[i] = nodeIdx; });
	else
	{
		m_pBVHData[childIdx.innerIdx()].setParent(nodeIdx.ToNative());
		m_pBVHData[childIdx.innerIdx()].setSibling(children(nodeIdx)[1 - localIdxToSetTo].ToNative());
	}
}

void e_BVHRebuilder::propagateBBChange(BVHIndex idx, const AABB& box, int localChildIdx)
{
	e_BVHNodeData* node = m_pBVHData + idx.innerIdx();
	if (localChildIdx == 0)
		node->setLeft(box);
	else node->setRight(box);
	BVHIndex p = BVHIndex::FromNative(m_pBVHData[idx.innerIdx()].getParent());
	if (p.isValid())
		propagateBBChange(p, node->getBox(), getChildIdxInLocal(p, idx));
}

void e_BVHRebuilder::swapChildren(BVHIndex idx, int localChildIdx, int localGrandChildIdx)
{
	BVHIndex childIdx = children(idx)[localChildIdx], otherChildIdx = children(idx)[1 - localChildIdx],
		grandChildIdx = children(otherChildIdx)[localGrandChildIdx], otherGrandChildIdx = children(otherChildIdx)[1 - localGrandChildIdx];

	setChild(otherChildIdx, childIdx, localGrandChildIdx);
	setChild(idx, grandChildIdx, localChildIdx);

	int grandChildLeafNum = numLeafs(grandChildIdx), childLeafNum = numLeafs(childIdx);
	BVHNodeInfo::changeCount(bvhNodeData, m_pBVHData, otherChildIdx, -grandChildLeafNum + childLeafNum);
	BVHNodeInfo::changeCount(bvhNodeData, m_pBVHData, idx, -childLeafNum + grandChildLeafNum);
}

void e_BVHRebuilder::enumerateLeaf(BVHIndex idx, const std::function<void(int)>& f)
{
	if (m_pBVHIndices)
	{
		int i = idx.leafIdx();
		do
		{
			f(m_pBVHIndices[i].getIndex());
		} while (!m_pBVHIndices[i++].getFlag());
	}
	else f(idx.leafIdx());
}

e_BVHRebuilder::BVHIndex e_BVHRebuilder::createLeaf(unsigned int nodeIdx, BVHIndex oldLeaf)
{
	if (m_pBVHIndices)
	{
		if (m_UBVHIndicesCount >= m_uBVHIndicesLength)
			throw std::runtime_error(__FUNCTION__);
		int leafIdx = m_UBVHIndicesCount;
		auto f = [&](int i)
		{
			m_pBVHIndices[m_UBVHIndicesCount].setIndex(i);
			m_pData->setObject(m_UBVHIndicesCount++, i);
		};
		f(nodeIdx);
		if (oldLeaf != BVHIndex::INVALID())
			enumerateLeaf(oldLeaf, f);
		m_pBVHIndices[m_UBVHIndicesCount - 1].setFlag(true);
		return BVHIndex::FromSceneNode(leafIdx);
	}
	else return BVHIndex::FromSceneNode(nodeIdx);
}

e_BVHRebuilder::BVHIndex e_BVHRebuilder::getLeafIdx(unsigned int objIdx)
{
	if (!m_pBVHIndices)
		return BVHIndex::FromSceneNode(objIdx);
	bool found = false;
	BVHIndexTuple c = children(nodeToBVHNode[objIdx]);
	enumerateLeaf(c[0], [&](int i){if (i == objIdx) found = true; });
	if (found)
		return c[0];
	enumerateLeaf(c[1], [&](int i){if (i == objIdx) found = true; });
	if (found)
		return c[0];
	throw std::runtime_error(__FUNCTION__);
}

bool e_BVHRebuilder::removeObjectFromLeaf(BVHIndex leafIdx, unsigned int objIdx)
{
	if (!m_pBVHIndices)
		throw std::runtime_error(__FUNCTION__);
	int i = leafIdx.leafIdx(), objCount = 0;
	bool moveMode = false;
	do
	{
		objCount++;
		if (m_pBVHIndices[i].getIndex() == objIdx)
			moveMode = true;
		if (moveMode)
			m_pBVHIndices[i - 1] = m_pBVHIndices[i];
	} while (!m_pBVHIndices[i++].getFlag());
	return objCount == 1;
}

void e_BVHRebuilder::writeGraphPart(BVHIndex idx, BVHIndex parent, std::ofstream& f)
{
	if (idx.isLeaf())
	{
		enumerateLeaf(idx, [&](int i){f << parent.innerIdx() << " -> " << i << "[style=dotted];\n"; });
	}
	else
	{
		if (parent.isValid())
			f << parent.innerIdx() << " -> " << idx.innerIdx() << ";\n";
		BVHIndexTuple c = children(idx);
		for (int i = 0; i < 2; i++)
			if (!c[i].NoNode())
				writeGraphPart(c[i], idx, f);
	}
}

void e_BVHRebuilder::printGraph(const std::string& path)
{
	std::ofstream file;
	file.open(path);
	file << "digraph SceneBVH {\nnode [fontname=\"Arial\"];\n";
	writeGraphPart(BVHIndex::FromNative(startNode), BVHIndex::INVALID(), file);
	file << "}";
	file.close();
}
