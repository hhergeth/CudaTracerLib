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
	int idx() const
	{
		if (isLeaf())
			return leafIdx();
		else return innerIdx();
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

	static void changeCount(std::vector<BVHNodeInfo>& nodeInfos, const e_BVHNodeData* nodeData, int idx, int off)
	{
		nodeInfos[idx].numLeafs += off;
		int parent = nodeData[idx].getParent();
		if (parent != -1)
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
		t->bvhNodeData.clear();
		t->nodeToBVHNode.clear();
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
		return pNode;
	}
	virtual void handleLastLeafObject()
	{
	}
	virtual void HandleStartNode(int startNode)
	{
		t->startNode = startNode;
	}
	int buildInfoTree(BVHIndex idx, BVHIndex parent)
	{
		BVHIndexTuple c = BVHIndexTuple::FromChildren(t->m_pBVHData + idx.innerIdx());
		int s = 0;
		for (int i = 0; i < 2; i++)
		{
			if (c[i].isLeaf())
			{
				t->nodeToBVHNode[c[i].leafIdx()] = idx;
				s++;
			}
			else if (!c[i].NoNode())
				s += buildInfoTree(c[i], idx);
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
	BVHNodeInfo::changeCount(bvhNodeData, m_pBVHData, nodeIdx.innerIdx(), -1);
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

void e_BVHRebuilder::insertNode(BVHIndex bvhNodeIdx, unsigned int nodeIdx, const AABB& nodeWorldBox)
{
	if (bvhNodeIdx.isLeaf())//split leaf
	{
		e_BVHNodeData* node = m_pBVHData + m_uBvhNodeCount++;
		BVHIndex parentIdx = nodeToBVHNode[bvhNodeIdx.leafIdx()];
		int localIdx = getChildIdxInLocal(parentIdx, bvhNodeIdx);
		BVHIndex idx = BVHIndex::FromBVHNode(node - m_pBVHData);
		setChild(idx, BVHIndex::FromSceneNode(nodeIdx), 0);
		setChild(idx, bvhNodeIdx, 1);
		bvhNodeData[idx.innerIdx()] = BVHNodeInfo(1);//only set one so we can increment
		setChild(parentIdx, idx, localIdx);
		BVHNodeInfo::changeCount(bvhNodeData, m_pBVHData, idx.innerIdx(), +1);
		nodeToBVHNode[nodeIdx] = idx;
		nodeToBVHNode[bvhNodeIdx.leafIdx()] = idx;
		propagateBBChange(parentIdx, node->getBox(), localIdx);
	}
	else
	{
		e_BVHNodeData* node = m_pBVHData + bvhNodeIdx.innerIdx();
		BVHIndexTuple c = children(bvhNodeIdx);
		if (c[0].NoNode() || c[1].NoNode())//insert into one child
		{
			node->setChild(c[1].NoNode(), ~nodeIdx, m_pData->getBox(nodeIdx));
			BVHNodeInfo::changeCount(bvhNodeData, m_pBVHData, bvhNodeIdx.innerIdx(), +1);
			nodeToBVHNode[nodeIdx] = bvhNodeIdx;
			propagateBBChange(bvhNodeIdx, nodeWorldBox, c[1].NoNode());
		}
		else
		{
			float left_sah, right_sah;
			sahModified(bvhNodeIdx, nodeWorldBox, left_sah, right_sah);
			insertNode(left_sah < right_sah ? c[0] : c[1], nodeIdx, nodeWorldBox);
		}
	}
}

void e_BVHRebuilder::recomputeNode(BVHIndex bvhNodeIdx, AABB& newBox)
{
	if (bvhNodeIdx.isLeaf())
	{
		newBox = getBox(bvhNodeIdx);
	}
	else
	{
		BVHNodeInfo& info = bvhNodeData[bvhNodeIdx.innerIdx()];
		e_BVHNodeData* node = m_pBVHData + bvhNodeIdx.innerIdx();
		BVHIndexTuple c = children(bvhNodeIdx);
		bool modified = false;
		for (int i = 0; i < 2; i++)
			if ((c[i].isLeaf() && flaggedSceneNodes[c[i].leafIdx()]) || (!c[i].isLeaf() && !c[i].NoNode() && flaggedBVHNodes[c[i].innerIdx()]))
			{
				modified = true;
				recomputeNode(c[i], newBox);
				if (i == 0)
					node->setLeft(newBox);
				else node->setRight(newBox);
			}

		if (modified && 0)
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
			if (sah_rots[best_rot] < sah(c[0], c[1]))
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
		//else throw std::runtime_error("Invalid flaggednodes!");

		newBox = getBox(bvhNodeIdx);
	}
}

e_BVHRebuilder::e_BVHRebuilder(e_BVHNodeData* data, unsigned int a_BVHNodeLength, unsigned int a_SceneNodeLength)
	: m_pBVHData(data), m_uBVHDataLength(a_BVHNodeLength), m_uBvhNodeCount(0), startNode(-1)
{
	nodeToBVHNode.resize(a_SceneNodeLength);
	bvhNodeData.resize(a_BVHNodeLength);
	flaggedBVHNodes.resize(a_BVHNodeLength);
	flaggedSceneNodes.resize(a_SceneNodeLength);
}

e_BVHRebuilder::~e_BVHRebuilder()
{

}

void e_BVHRebuilder::invalidateNode(unsigned int n)
{
	if (nodesToInsert.find(n) == nodesToInsert.end())
		nodesToRecompute.insert(n);
}

void e_BVHRebuilder::addNode(unsigned int n)
{
	nodesToInsert.insert(n);
}

void e_BVHRebuilder::removeNode(unsigned int n)
{
	nodesToRemove.insert(n);
	std::set<unsigned int>::iterator it = nodesToInsert.find(n), it2 = nodesToRecompute.find(n);
	if (it != nodesToInsert.end())
		nodesToInsert.erase(it);
	if (it2 != nodesToRecompute.end())
		nodesToRecompute.erase(it2);
}

bool e_BVHRebuilder::needsBuild()
{
	return nodesToRecompute.size() != 0 || nodesToInsert.size() != 0 || nodesToRemove.size() != 0;
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
	int lN = c[0].isLeaf() ? 1 : bvhNodeData[c[0].innerIdx()].numLeafs;
	int rN = c[1].isLeaf() ? 1 : bvhNodeData[c[1].innerIdx()].numLeafs;
	leftSAH = lAd * (lN + 1) + rA * rN;
	rightSAH = lA * lN + rAd * (rN + 1);
}

int e_BVHRebuilder::validateTree(BVHIndex idx, BVHIndex parent)
{
	if (idx.isLeaf())
	{
		if (nodeToBVHNode[idx.leafIdx()] != parent)
			throw std::runtime_error(__FUNCTION__);
		return 1;
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
		return s;
	}
}

AABB e_BVHRebuilder::getBox(BVHIndex idx)
{
	if (idx.isLeaf())
		return m_pData->getBox(idx.leafIdx());
	else return m_pBVHData[idx.innerIdx()].getBox();
}

void e_BVHRebuilder::propagateFlag(BVHIndex idx)
{
	if (idx.isLeaf())
	{
		flaggedSceneNodes[idx.leafIdx()] = 1;
		propagateFlag(nodeToBVHNode[idx.leafIdx()]);
	}
	else
	{
		flaggedBVHNodes[idx.innerIdx()] = 1;
		BVHIndex parent = BVHIndex::FromNative(m_pBVHData[idx.innerIdx()].getParent());
		if (parent.isValid())
			propagateFlag(parent);
	}
}

int e_BVHRebuilder::numberGrandchildren(BVHIndex idx, int localChildIdx)
{
	if (idx.isLeaf())
		throw std::runtime_error(__FUNCTION__);
	BVHIndexTuple c = children(idx);
	BVHIndex child = c[localChildIdx];
	if (child.isLeaf())
		return 0;
	BVHIndexTuple c2 = children(child);
	return (!c2[0].NoNode()) + (!c2[1].NoNode());
}

float e_BVHRebuilder::sah(BVHIndex lhs, BVHIndex rhs)
{
	AABB lBox = getBox(lhs), rBox = getBox(rhs);
	int lNum = lhs.isLeaf() ? 1 : bvhNodeData[lhs.innerIdx()].numLeafs;
	int rNum = rhs.isLeaf() ? 1 : bvhNodeData[rhs.innerIdx()].numLeafs;
	return lBox.Area() * lNum + rBox.Area() * rNum;
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
	return idx.isLeaf() ? 1 : bvhNodeData[idx.innerIdx()].numLeafs;
}

void e_BVHRebuilder::setChild(BVHIndex nodeIdx, BVHIndex childIdx, int localIdxToSetTo)
{
	m_pBVHData[nodeIdx.innerIdx()].setChild(localIdxToSetTo, childIdx.ToNative(), getBox(childIdx));
	if (childIdx.isLeaf())
		nodeToBVHNode[childIdx.leafIdx()] = nodeIdx;
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
	BVHNodeInfo::changeCount(bvhNodeData, m_pBVHData, otherChildIdx.innerIdx(), -grandChildLeafNum + childLeafNum);
	BVHNodeInfo::changeCount(bvhNodeData, m_pBVHData, idx.innerIdx(), -childLeafNum + grandChildLeafNum);
}