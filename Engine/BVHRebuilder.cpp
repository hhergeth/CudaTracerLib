#include <StdAfx.h>
#include "Buffer.h"
#include "BVHRebuilder.h"
#include "SceneBuilder/SplitBVHBuilder.hpp"
#include "Mesh.h"
#include <algorithm>

namespace CudaTracerLib {

#define NO_NODE 0x76543210

//bvh tree rotations from
//http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.331.9382&rep=rep1&type=pdf

struct BVHRebuilder::BVHIndex
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
	bool operator<(const BVHIndex& rhs) const
	{
		return native < rhs.native;
	}
	static BVHIndex INVALID()
	{
		return FromNative(-1);
	}
};

struct BVHRebuilder::BVHIndexTuple
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

	static BVHIndexTuple FromChildren(const BVHNodeData* n)
	{
		BVHIndexTuple t;
		for (int i = 0; i < 2; i++)
			t.values[i] = BVHIndex::FromNative(n->getChildren()[i]);
		return t;
	}
};

struct BVHRebuilder::BVHNodeInfo
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

	static void changeCount(std::vector<BVHNodeInfo>& nodeInfos, const BVHNodeData* nodeData, BVHIndex idx, int off)
	{
		nodeInfos[idx.innerIdx()].numLeafs += off;
		BVHIndex parent = BVHIndex::FromNative(nodeData[idx.innerIdx()].getParent());
		if (parent.isValid())
			changeCount(nodeInfos, nodeData, parent, off);
	}
};

class BVHRebuilder::BuilderCLB : public IBVHBuilderCallback
{
	BVHRebuilder* t;
public:
	BuilderCLB(BVHRebuilder* c)
		: t(c)
	{
		t->m_uBvhNodeCount = 0;
		t->m_UBVHIndicesCount = 0;
	}
	virtual void iterateObjects(std::function<void(unsigned int)> f)
	{
		t->m_pData->iterateObjects(f);
	}
	virtual void getBox(unsigned int index, AABB* out) const
	{
		*out = t->m_pData->getBox(index);
	}
	virtual void HandleBoundingBox(const AABB& box)
	{

	}
	virtual BVHNodeData* HandleNodeAllocation(int* index)
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
};

void BVHRebuilder::removeNodeAndCollapse(BVHIndex nodeIdx, BVHIndex childIdx)
{
	int childIdxLocal = getChildIdxInLocal(nodeIdx, childIdx);
	BVHNodeData* node = m_pBVHData + nodeIdx.innerIdx();
	BVHIndexTuple children = this->children(nodeIdx);
	node->setChild(childIdxLocal, NO_NODE, AABB::Identity());
	if (!children[1 - childIdxLocal].NoNode())
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
		else
		{
			int parentLocal = getChildIdxInLocal(grandpaIdx, nodeIdx);
			setChild(grandpaIdx, otherChild, parentLocal, nodeIdx);
			propagateBBChange(grandpaIdx, getBox(nodeIdx), parentLocal);
		}
	}
}

void BVHRebuilder::insertNode(BVHIndex bvhNodeIdx, BVHIndex parent, unsigned int nodeIdx, const AABB& nodeWorldBox)
{
	if (bvhNodeIdx.isLeaf())
	{
		if (m_pBVHIndices)//add to leaf
		{
			//TODO sah for split?
			BVHNodeInfo::changeCount(bvhNodeData, m_pBVHData, parent, +1);
			BVHIndex newLeafIdx = createLeaf(nodeIdx, bvhNodeIdx);
			setChild(parent, newLeafIdx, getChildIdxInLocal(parent, bvhNodeIdx), bvhNodeIdx);
		}
		else//split leaf
		{
			BVHNodeData* node = m_pBVHData + m_uBvhNodeCount++;
			int localIdx = getChildIdxInLocal(parent, bvhNodeIdx);
			BVHIndex idx = BVHIndex::FromBVHNode(node - m_pBVHData);
			setChild(parent, idx, localIdx, BVHIndex::INVALID(), false);
			setChild(idx, BVHIndex::FromSceneNode(nodeIdx), 0, BVHIndex::INVALID(), false);
			setChild(idx, bvhNodeIdx, 1, parent, true);
			bvhNodeData[idx.innerIdx()] = BVHNodeInfo(1);//only set one so we can increment
			BVHNodeInfo::changeCount(bvhNodeData, m_pBVHData, idx, +1);
			objectToBVHNodes[nodeIdx].clear();
			objectToBVHNodes[nodeIdx].push_back(idx);
			propagateBBChange(parent, node->getBox(), localIdx);
		}
	}
	else
	{
		BVHNodeData* node = m_pBVHData + bvhNodeIdx.innerIdx();
		BVHIndexTuple c = children(bvhNodeIdx);
		if (c[0].NoNode() || c[1].NoNode())//insert into one child
		{
			node->setChild(c[1].NoNode(), createLeaf(nodeIdx, BVHIndex::INVALID()).ToNative(), m_pData->getBox(nodeIdx));
			BVHNodeInfo::changeCount(bvhNodeData, m_pBVHData, bvhNodeIdx, +1);
			objectToBVHNodes[nodeIdx].clear();
			objectToBVHNodes[nodeIdx].push_back(bvhNodeIdx);
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

void BVHRebuilder::recomputeNode(BVHIndex bvhNodeIdx, AABB& newBox)
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
		BVHNodeData* node = m_pBVHData + bvhNodeIdx.innerIdx();
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
		//else throw std::runtime_error("Invalid flaggednodes!");

		newBox = getBox(bvhNodeIdx);
	}
}

void BVHRebuilder::propagateFlag(BVHIndex idx)
{
	if (idx.isLeaf())
	{
		int objIdx = m_pBVHIndices ? m_pBVHIndices[idx.leafIdx()].getIndex() : idx.leafIdx();
		auto& nodes = objectToBVHNodes[objIdx];
		for (size_t i = 0; i < nodes.size(); i++)
			propagateFlag(nodes[i]);
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

void BVHRebuilder::SetEmpty()
{
	startNode = -1;
	m_uBvhNodeCount = 0;
}

bool BVHRebuilder::Build(ISpatialInfoProvider* data, bool invalidateAll)
{
	this->m_pData = data;
	bool modified = false;
	if (needsBuild() || invalidateAll)
	{
		modified = true;
		if (startNode == -1)
		{
			Platform::SetMemory(m_pBVHData, sizeof(BVHNodeData) * m_uBVHDataLength);
			Platform::SetMemory(m_pBVHIndices, sizeof(TriIntersectorData2) * m_uBVHIndicesLength);
			BuilderCLB b(this);
			SplitBVHBuilder::Platform Pq;
			if (!m_pBVHIndices)
				Pq.m_maxLeafSize = 1;
			SplitBVHBuilder bu(&b, Pq, SplitBVHBuilder::BuildParams());
			bu.run();
			BuildInfoTree(BVHIndex::FromNative(startNode), BVHIndex::INVALID());
#ifndef NDEBUG
			validateTree(BVHIndex::FromNative(startNode), BVHIndex::INVALID());
#endif
		}
		else
		{
			typedef std::set<unsigned int>::iterator n_it;
			for (n_it it = nodesToRemove.begin(); it != nodesToRemove.end(); ++it)
			{
				std::vector<BVHIndex> leafIndices;
				const auto& nodes = objectToBVHNodes[*it];

				if (!m_pBVHIndices)
					leafIndices.push_back(BVHIndex::FromSceneNode(*it));
				else
				{
					for (size_t i = 0; i < nodes.size(); i++)
					{
						bool found = false;
						BVHIndexTuple c = children(nodes[i]);
						enumerateLeaf(c[0], [&](int i){if (i == *it) found = true; });
						if (found)
							leafIndices.push_back(c[0]);
						else
						{
							enumerateLeaf(c[1], [&](int i){if (i == *it) found = true; });
							if (found)
								leafIndices.push_back(c[0]);
							else std::runtime_error(__FUNCTION__);
						}
					}
				}

				for (size_t i = 0; i < leafIndices.size(); i++)
					if (!m_pBVHIndices || removeObjectFromLeaf(leafIndices[i], *it))
						removeNodeAndCollapse(nodes[i], leafIndices[i]);
			}
			for (n_it it = nodesToInsert.begin(); it != nodesToInsert.end(); ++it)
			{
				objectToBVHNodes[*it].clear();
				insertNode(BVHIndex::FromNative(startNode), BVHIndex::INVALID(), *it, data->getBox(*it));
			}
			recomputeAll = invalidateAll;
			if (!recomputeAll)
				m_pData->iterateObjects([&](unsigned int i)
			{
				if (nodesToRecompute[i])
					propagateFlag(BVHIndex::FromSceneNode((unsigned int)i));
			});
			AABB box;
			if (recomputeAll || flaggedBVHNodes[startNode / 4])
				recomputeNode(BVHIndex::FromNative(startNode), box);
			flaggedBVHNodes.reset();
		}
		//printGraph("1.txt");
#ifndef NDEBUG
		validateTree(BVHIndex::FromNative(startNode), BVHIndex::INVALID());
#endif
	}
	nodesToRecompute.reset();
	nodesToInsert.clear();
	nodesToRemove.clear();
	m_uModifiedCount = 0;
	return modified;
}

BVHRebuilder::BVHRebuilder(BVHNodeData* data, unsigned int a_BVHNodeLength, unsigned int a_SceneNodeLength, TriIntersectorData2* indices, unsigned int a_IndicesLength)
	: m_pBVHData(data), m_uBVHDataLength(a_BVHNodeLength), m_uBvhNodeCount(0), startNode(-1), m_pBVHIndices(indices), m_uBVHIndicesLength(a_IndicesLength), m_UBVHIndicesCount(0),
	m_uModifiedCount(0), m_pData(0)
{
	if (a_SceneNodeLength > MAX_NODES)
		throw std::runtime_error("BVHRebuilder too many objects!");
	objectToBVHNodes.resize(a_SceneNodeLength);
	bvhNodeData.resize(m_uBVHDataLength);
}

BVHRebuilder::BVHRebuilder(Mesh* mesh, ISpatialInfoProvider* data)
	: m_pBVHData(mesh->m_sNodeInfo(0)), m_uBVHDataLength(mesh->m_sNodeInfo.getLength()), m_uBvhNodeCount(0), startNode(-1),
	m_pBVHIndices(mesh->m_sIndicesInfo(0)), m_uBVHIndicesLength(mesh->m_sIndicesInfo.getLength()), m_UBVHIndicesCount(0),
	m_uModifiedCount(0), m_pData(data)
{
	unsigned int a_SceneNodeLength = mesh->m_sTriInfo.getLength();
	if (a_SceneNodeLength > MAX_NODES)
		throw std::runtime_error("BVHRebuilder too many objects!");
	objectToBVHNodes.resize(a_SceneNodeLength);
	bvhNodeData.resize(m_uBVHDataLength);

	m_uBvhNodeCount = m_uBVHDataLength;
	m_UBVHIndicesCount = m_uBVHIndicesLength;
	startNode = 0;
	BuilderCLB b(this);
	BuildInfoTree(BVHIndex::FromNative(startNode), BVHIndex::INVALID());
	validateTree(BVHIndex::FromNative(startNode), BVHIndex::INVALID(), false);//at that point the aabb's are not yet calculated so just skip checking them
	AABB box;
	recomputeAll = true;
	recomputeNode(BVHIndex::FromNative(startNode), box);
	validateTree(BVHIndex::FromNative(startNode), BVHIndex::INVALID());
	m_pData = 0;
	recomputeAll = false;
}

BVHRebuilder::~BVHRebuilder()
{

}

int BVHRebuilder::BuildInfoTree(BVHIndex idx, BVHIndex parent)
{
	BVHIndexTuple c = BVHIndexTuple::FromChildren(m_pBVHData + idx.innerIdx());
	int s = 0;
	for (int i = 0; i < 2; i++)
	{
		if (c[i].isLeaf())
		{
			enumerateLeaf(c[i], [&](int j)
			{
				objectToBVHNodes[j].push_back(idx);
				s++;
			});
		}
		else if (!c[i].NoNode())
			s += BuildInfoTree(c[i], idx);
	}
	bvhNodeData[idx.innerIdx()].numLeafs = s;
	return s;
}

void BVHRebuilder::invalidateNode(unsigned int n)
{
	m_uModifiedCount++;
	if (nodesToInsert.find(n) == nodesToInsert.end())
		nodesToRecompute.set(n);
}

void BVHRebuilder::addNode(unsigned int n)
{
	nodesToInsert.insert(n);
}

void BVHRebuilder::removeNode(unsigned int n)
{
	nodesToRemove.insert(n);
	std::set<unsigned int>::iterator it = nodesToInsert.find(n);
	if (it != nodesToInsert.end())
		nodesToInsert.erase(it);
	nodesToRecompute.reset(n);
}

bool BVHRebuilder::needsBuild()
{
	return m_uModifiedCount || nodesToInsert.size() != 0 || nodesToRemove.size() != 0;
}

AABB BVHRebuilder::getBox()
{
	return m_pBVHData[startNode].getBox();
}

int BVHRebuilder::getChildIdxInLocal(BVHIndex nodeIdx, BVHIndex childIdx)
{
	BVHIndexTuple c = children(nodeIdx);
	if (c[0] != childIdx && c[1] != childIdx)
		throw std::runtime_error("Invalid tree passed!");
	return c[1] == childIdx;
}

void BVHRebuilder::sahModified(BVHIndex nodeIdx, const AABB& box, float& leftSAH, float& rightSAH)
{
	AABB left, right;
	BVHNodeData* node = m_pBVHData + nodeIdx.innerIdx();
	node->getBox(left, right);
	BVHIndexTuple c = children(nodeIdx);
	float lA = left.Area(), rA = right.Area();
	left = left.Extend(box);
	right = right.Extend(box);
	float lAd = left.Area(), rAd = right.Area();
	int lN = numLeafs(c[0]), rN = numLeafs(c[1]);
	leftSAH = lAd * (lN + 1) + rA * rN;
	rightSAH = lA * lN + rAd * (rN + 1);
}

int BVHRebuilder::validateTree(BVHIndex idx, BVHIndex parent, bool checkAABB)
{
	if (idx.isLeaf())
	{
		enumerateLeaf(idx, [&](int i)
		{
			const auto& nodes = objectToBVHNodes[i];//local copy, sort shouldn't matter but still
			if (std::find(nodes.begin(), nodes.end(), parent) == nodes.end())
				throw std::runtime_error(__FUNCTION__);
		});
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
		int s = validateTree(c[0], idx, checkAABB) + validateTree(c[1], idx, checkAABB);
		if (s != info.numLeafs)
			throw std::runtime_error(__FUNCTION__);
		if (checkAABB)
		{
			AABB leftBox = getBox(c[0]), rightBox = getBox(c[1]), box = getBox(idx), box2 = leftBox.Extend(rightBox);
			if (!box.Contains(box2.minV) || !box.Contains(box2.maxV))
				throw std::runtime_error(__FUNCTION__);
		}
		return s;
	}
}

AABB BVHRebuilder::getBox(BVHIndex idx)
{
	if (idx.NoNode())
		return AABB::Identity();
	if (idx.isLeaf())
	{
		AABB box = AABB::Identity();
		enumerateLeaf(idx, [&](int i){box = box.Extend(m_pData->getBox(i)); });
		return box;
	}
	else return m_pBVHData[idx.innerIdx()].getBox();
}

int BVHRebuilder::numberGrandchildren(BVHIndex idx, int localChildIdx)
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

float BVHRebuilder::sah(BVHIndex idx, int localChildIdx, int localGrandChildIdx)
{
	BVHIndex childIdx = children(idx)[localChildIdx], otherChildIdx = children(idx)[1 - localChildIdx],
		grandChildIdx = children(otherChildIdx)[localGrandChildIdx], otherGrandChildIdx = children(otherChildIdx)[1 - localGrandChildIdx];

	float childRhsArea = getBox(grandChildIdx).Area();
	int childRhsNum = numLeafs(grandChildIdx);

	AABB boxLhs1 = getBox(childIdx), boxLhs2 = getBox(otherGrandChildIdx);
	int numLhs1 = numLeafs(childIdx), numLhs2 = numLeafs(otherGrandChildIdx);
	boxLhs1 = boxLhs1.Extend(boxLhs2);

	return boxLhs1.Area() * (numLhs1 + numLhs2) + childRhsArea * childRhsNum;
}

BVHRebuilder::BVHIndexTuple BVHRebuilder::children(BVHIndex idx)
{
	return BVHIndexTuple::FromChildren(m_pBVHData + idx.innerIdx());
}

int BVHRebuilder::numLeafs(BVHIndex idx)
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

void BVHRebuilder::setChild(BVHIndex nodeIdx, BVHIndex childIdx, int localIdxToSetTo, BVHIndex oldParent, bool prop)
{
	m_pBVHData[nodeIdx.innerIdx()].setChild(localIdxToSetTo, childIdx.ToNative(), getBox(childIdx));
	if (childIdx.isLeaf())
		enumerateLeaf(childIdx, [&](int i)
	{
		auto& nodes = objectToBVHNodes[i];
		auto it = std::find(nodes.begin(), nodes.end(), oldParent);
		if (oldParent != BVHIndex::INVALID() && it == nodes.end())
			throw std::runtime_error(__FUNCTION__);
		if (it == nodes.end())
			objectToBVHNodes[i].push_back(nodeIdx);
		else *it = nodeIdx;
	});
	else
	{
		m_pBVHData[childIdx.innerIdx()].setParent(nodeIdx.ToNative());
		m_pBVHData[childIdx.innerIdx()].setSibling(children(nodeIdx)[1 - localIdxToSetTo].ToNative());
	}
	propagateBBChange(nodeIdx, getBox(childIdx), localIdxToSetTo);
}

void BVHRebuilder::propagateBBChange(BVHIndex idx, const AABB& box, int localChildIdx)
{
	BVHNodeData* node = m_pBVHData + idx.innerIdx();
	if (localChildIdx == 0)
		node->setLeft(box);
	else node->setRight(box);
	BVHIndex p = BVHIndex::FromNative(m_pBVHData[idx.innerIdx()].getParent());
	if (p.isValid())
		propagateBBChange(p, node->getBox(), getChildIdxInLocal(p, idx));
}

void BVHRebuilder::swapChildren(BVHIndex idx, int localChildIdx, int localGrandChildIdx)
{
	BVHIndex childIdx = children(idx)[localChildIdx], otherChildIdx = children(idx)[1 - localChildIdx],
		grandChildIdx = children(otherChildIdx)[localGrandChildIdx], otherGrandChildIdx = children(otherChildIdx)[1 - localGrandChildIdx];

	setChild(otherChildIdx, childIdx, localGrandChildIdx, idx);
	setChild(idx, grandChildIdx, localChildIdx, otherChildIdx);

	int grandChildLeafNum = numLeafs(grandChildIdx), childLeafNum = numLeafs(childIdx);
	BVHNodeInfo::changeCount(bvhNodeData, m_pBVHData, otherChildIdx, -grandChildLeafNum + childLeafNum);
	BVHNodeInfo::changeCount(bvhNodeData, m_pBVHData, idx, -childLeafNum + grandChildLeafNum);
}

void BVHRebuilder::enumerateLeaf(BVHIndex idx, const std::function<void(int)>& f)
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

BVHRebuilder::BVHIndex BVHRebuilder::createLeaf(unsigned int nodeIdx, BVHIndex oldLeaf)
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

bool BVHRebuilder::removeObjectFromLeaf(BVHIndex leafIdx, unsigned int objIdx)
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

void BVHRebuilder::writeGraphPart(BVHIndex idx, BVHIndex parent, std::ofstream& f)
{
	if (idx.isLeaf())
	{
		enumerateLeaf(idx, [&](int i){f << parent.ToNative() << " -> " << ~i << "[style=dotted];\n"; });
	}
	else
	{
		if (parent.isValid())
			f << parent.ToNative() << " -> " << idx.ToNative() << ";\n";
		BVHIndexTuple c = children(idx);
		for (int i = 0; i < 2; i++)
			if (!c[i].NoNode())
				writeGraphPart(c[i], idx, f);
	}
}

void BVHRebuilder::printGraph(const std::string& path)
{
	std::ofstream file;
	file.open(path);
	file << "digraph SceneBVH {\nnode [fontname=\"Arial\"];\n";
	writeGraphPart(BVHIndex::FromNative(startNode), BVHIndex::INVALID(), file);
	file << "}";
	file.close();
}

}
