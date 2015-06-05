#include "StdAfx.h"
#include "e_SceneBVH.h"
#include "e_Mesh.h"
#include "e_Node.h"
#include "SceneBuilder/SplitBVHBuilder.hpp"

#define NO_NODE 0x76543210

//bvh tree rotations from
//http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.331.9382&rep=rep1&type=pdf

struct e_SceneBVH::BVHIndex
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
	static BVHIndex FromSceneNode(e_StreamReference(e_Node) n)
	{
		return BVHIndex(~n.getIndex());
	}
	static BVHIndex FromBVHNode(e_StreamReference(e_BVHNodeData) n)
	{
		return BVHIndex(n.getIndex() * 4);
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
};

const static e_SceneBVH::BVHIndex INVALID = e_SceneBVH::BVHIndex::FromNative(-1);

struct e_SceneBVH::BVHIndexTuple
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
};

e_SceneBVH::BVHIndexTuple FromChildren(const e_BVHNodeData* n)
{
	e_SceneBVH::BVHIndexTuple t;
	for (int i = 0; i < 2; i++)
		t.values[i] = e_SceneBVH::BVHIndex::FromNative(n->getChildren()[i]);
	return t;
}

struct e_SceneBVH::BVHNodeInfo
{
	BVHIndex parent;
	int numLeafs;

	BVHNodeInfo()
		: numLeafs(0)
	{

	}

	BVHNodeInfo(BVHIndex p, int l)
		: parent(p), numLeafs(l)
	{

	}

	void changeCount(std::vector<BVHNodeInfo>& nodeInfos, int off)
	{
		numLeafs += off;
		if (parent.isValid())
			nodeInfos[parent.innerIdx()].changeCount(nodeInfos, off);
	}
};

struct e_SceneBVH::SceneInfo
{
	e_StreamReference(e_Node) a_Nodes;
	e_BufferReference<e_Mesh, e_KernelMesh> a_Meshes;

	SceneInfo(e_StreamReference(e_Node) a, e_BufferReference<e_Mesh, e_KernelMesh> b)
		: a_Nodes(a), a_Meshes(b)
	{

	}
};

class e_SceneBVH::BuilderCLB : public IBVHBuilderCallback
{
	e_SceneBVH* t;
public:
	BuilderCLB(e_SceneBVH* c)
		: t(c)
	{
		t->m_sBvhNodeCount = 0;
	}
	virtual unsigned int Count() const
	{
		return t->info->a_Nodes.getLength();
	}
	virtual void getBox(unsigned int index, AABB* out) const
	{
		*out = t->getWorldNodeBox(t->info->a_Nodes(index));
	}
	virtual void HandleBoundingBox(const AABB& box)
	{
		t->m_sBox = box;
	}
	virtual e_BVHNodeData* HandleNodeAllocation(int* index)
	{
		*index = t->m_sBvhNodeCount * 4;
		return t->m_pNodes->operator()(t->m_sBvhNodeCount++);
	}
	virtual void setSibling(int idx, int sibling)
	{
		if (idx >= 0)
			t->m_pNodes->operator()(idx / 4)->setSibling(sibling);
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
		BVHIndexTuple c = FromChildren(t->m_pNodes->operator()(idx.innerIdx()));
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
		t->bvhNodeData[idx.innerIdx()].parent = parent;
		t->bvhNodeData[idx.innerIdx()].numLeafs = s;
		return s;
	}
};

void e_SceneBVH::setChild(BVHIndex nodeIdx, BVHIndex childIdx, int localIdxToSetTo)
{
	if (childIdx.isLeaf())
		nodeToBVHNode[childIdx.leafIdx()] = nodeIdx;
	else bvhNodeData[childIdx.innerIdx()].parent = nodeIdx;
	m_pNodes->operator()(nodeIdx.innerIdx())->setChild(localIdxToSetTo, childIdx.ToNative(), getBox(childIdx));
}

void e_SceneBVH::propagateBBChange(BVHIndex idx, const AABB& box, int localChildIdx)
{
	e_BVHNodeData* node = m_pNodes->operator()(idx.innerIdx());
	if (localChildIdx == 0)
		node->setLeft(box);
	else node->setRight(box);
	BVHIndex p = bvhNodeData[idx.innerIdx()].parent;
	if (p.isValid())
		propagateBBChange(p, node->getBox(), getChildIdxInLocal(p, idx));
	else m_sBox = node->getBox();
}

void e_SceneBVH::swapChildren(BVHIndex idx, int localChildIdx, int localGrandChildIdx)
{
	BVHIndex childIdx = children(idx)[localChildIdx], otherChildIdx = children(idx)[1 - localChildIdx],
			 grandChildIdx = children(otherChildIdx)[localGrandChildIdx], otherGrandChildIdx = children(otherChildIdx)[1 - localGrandChildIdx];

	setChild(otherChildIdx, childIdx, localGrandChildIdx);
	setChild(idx, grandChildIdx, localChildIdx);

	int grandChildLeafNum = numLeafs(grandChildIdx), childLeafNum = numLeafs(childIdx);
	bvhNodeData[otherChildIdx.innerIdx()].changeCount(bvhNodeData, -grandChildLeafNum + childLeafNum);
	bvhNodeData[idx.innerIdx()].changeCount(bvhNodeData, -childLeafNum + grandChildLeafNum);
}

void e_SceneBVH::removeNodeAndCollapse(BVHIndex nodeIdx, BVHIndex childIdx)
{
	int childIdxLocal = getChildIdxInLocal(nodeIdx, childIdx);
	e_BVHNodeData* node = m_pNodes->operator()(nodeIdx.innerIdx());
	BVHIndexTuple children = FromChildren(node);
	node->setChild(childIdxLocal, NO_NODE, AABB::Identity());
	propagateBBChange(nodeIdx, AABB::Identity(), childIdxLocal);
	bvhNodeData[nodeIdx.innerIdx()].changeCount(bvhNodeData , - 1);
	BVHIndex grandpaIdx = bvhNodeData[nodeIdx.innerIdx()].parent;
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
				m_sBox = m_pNodes->operator()(otherChild.innerIdx())->getBox();
				bvhNodeData[otherChild.innerIdx()].parent = INVALID;
			}
			else if (otherChild.isLeaf())
			{
				m_sBox = getWorldNodeBox(info->a_Nodes(otherChild.leafIdx()));
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

void e_SceneBVH::insertNode(BVHIndex bvhNodeIdx, unsigned int nodeIdx, const AABB& nodeWorldBox)
{
	if (bvhNodeIdx.isLeaf())//split leaf
	{
		e_StreamReference(e_BVHNodeData) node = m_pNodes->operator()(m_sBvhNodeCount++);
		BVHIndex parentIdx = nodeToBVHNode[bvhNodeIdx.leafIdx()];
		int localIdx = getChildIdxInLocal(parentIdx, bvhNodeIdx);
		BVHIndex idx = BVHIndex::FromBVHNode(node);
		setChild(idx, BVHIndex::FromSceneNode(info->a_Nodes(nodeIdx)), 0);
		setChild(idx, bvhNodeIdx, 1);
		bvhNodeData[idx.innerIdx()] = BVHNodeInfo(parentIdx, 1);//only set one so we can increment
		setChild(parentIdx, idx, localIdx);
		bvhNodeData[idx.innerIdx()].changeCount(bvhNodeData, +1);
		nodeToBVHNode[nodeIdx] = idx;
		nodeToBVHNode[bvhNodeIdx.leafIdx()] = idx;
		propagateBBChange(parentIdx, node->getBox(), localIdx);
	}
	else
	{
		e_BVHNodeData* node = m_pNodes->operator()(bvhNodeIdx.innerIdx());
		BVHIndexTuple c = FromChildren(node);
		if (c[0].NoNode() || c[1].NoNode())//insert into one child
		{
			node->setChild(c[1].NoNode(), ~nodeIdx, getWorldNodeBox(info->a_Nodes(nodeIdx)));
			bvhNodeData[bvhNodeIdx.innerIdx()].changeCount(bvhNodeData, +1);
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

void e_SceneBVH::recomputeNode(BVHIndex bvhNodeIdx, AABB& newBox)
{
	if (bvhNodeIdx.isLeaf())
	{
		newBox = getBox(bvhNodeIdx);
	}
	else
	{
		BVHNodeInfo& info = bvhNodeData[bvhNodeIdx.innerIdx()];
		e_BVHNodeData* node = m_pNodes->operator()(bvhNodeIdx.innerIdx());
		BVHIndexTuple c = FromChildren(node);
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

bool e_SceneBVH::Build(e_StreamReference(e_Node) a_Nodes, e_BufferReference<e_Mesh, e_KernelMesh> a_Meshes)
{
	SceneInfo lInfo(a_Nodes, a_Meshes);
	info = &lInfo;
	bool modified = false;
	if (a_Nodes.getLength() != 0 && needsBuild())
	{
		modified = true;
		size_t m = max(nodesToInsert.size(), nodesToRecompute.size(), nodesToRemove.size());
		if (m > a_Nodes.getLength() / 2 || startNode == -1)
		{
			BuilderCLB b(this);
			SplitBVHBuilder::Platform Pq;
			Pq.m_maxLeafSize = 1;
			SplitBVHBuilder bu(&b, Pq, SplitBVHBuilder::BuildParams());
			bu.run();
			b.buildInfoTree(BVHIndex::FromNative(startNode), INVALID);
			validateTree(BVHIndex::FromNative(startNode), INVALID);
		}
		else
		{
			typedef std::set<e_StreamReference(e_Node)>::iterator n_it;
			for (n_it it = nodesToRemove.begin(); it != nodesToRemove.end(); ++it)
				removeNodeAndCollapse(nodeToBVHNode[it->getIndex()], BVHIndex::FromSceneNode(*it));
			for (n_it it = nodesToInsert.begin(); it != nodesToInsert.end(); ++it)
				insertNode(BVHIndex::FromNative(startNode), it->getIndex(), getWorldNodeBox(*it));
			for (n_it it = nodesToRecompute.begin(); it != nodesToRecompute.end(); ++it)
				propagateFlag(BVHIndex::FromSceneNode(*it));
			if (flaggedBVHNodes[startNode / 4])
				recomputeNode(BVHIndex::FromNative(startNode), m_sBox);
			flaggedBVHNodes.assign(flaggedBVHNodes.size(), 0);
			flaggedSceneNodes.assign(flaggedSceneNodes.size(), 0);
			validateTree(BVHIndex::FromNative(startNode), INVALID);
		}
		//printGraph("1.txt", a_Nodes);
	}
	if (!a_Nodes.getLength())
	{
		startNode = -1;
		m_sBvhNodeCount = 0;
	}
	m_pNodes->Invalidate();
	m_pNodes->UpdateInvalidated();
	m_pTransforms->Invalidate();
	m_pTransforms->UpdateInvalidated();
	m_pInvTransforms->Invalidate();
	m_pInvTransforms->UpdateInvalidated();
	nodesToRecompute.clear();
	nodesToInsert.clear();
	nodesToRemove.clear();
	return modified;
}

e_SceneBVH::e_SceneBVH(unsigned int a_NodeCount)
	: startNode(-1), m_sBvhNodeCount(0)
{
	m_pNodes = new e_Stream<e_BVHNodeData>(a_NodeCount * 2);//largest binary tree has the same amount of inner nodes
	m_pTransforms = new e_Stream<float4x4>(a_NodeCount);
	m_pInvTransforms = new e_Stream<float4x4>(a_NodeCount);
	startNode = -1;
	m_sBox = AABB::Identity();
	for(unsigned int i = 0; i < a_NodeCount; i++)
	{
		*m_pTransforms[0](i).operator->() = *m_pInvTransforms[0](i).operator->() = float4x4::Identity();
	}
	//INVALIDATE
	m_pTransforms->malloc(m_pTransforms->getLength());
	m_pInvTransforms->malloc(m_pInvTransforms->getLength());
	m_pNodes->malloc(m_pNodes->getLength());
	nodeToBVHNode.resize(a_NodeCount);
	bvhNodeData.resize(m_pNodes->getLength());
	flaggedBVHNodes.resize(m_pNodes->getLength());
	flaggedSceneNodes.resize(a_NodeCount);
}

void e_SceneBVH::invalidateNode(e_BufferReference<e_Node, e_Node> n)
{
	nodesToRecompute.insert(n);
}

void e_SceneBVH::setTransform(e_BufferReference<e_Node, e_Node> n, const float4x4& mat)
{
	unsigned int nodeIdx = n.getIndex();
	*m_pTransforms[0](nodeIdx).operator->() = mat;
	*m_pInvTransforms[0](nodeIdx).operator->() = mat.inverse();
	m_pTransforms->Invalidate(nodeIdx, 1);
	m_pInvTransforms->Invalidate(nodeIdx, 1);
	invalidateNode(n);
}

e_SceneBVH::~e_SceneBVH()
{
	delete m_pNodes;
	delete m_pTransforms;
	delete m_pInvTransforms;
}

e_KernelSceneBVH e_SceneBVH::getData(bool devicePointer)
{
	e_KernelSceneBVH q;
	q.m_pNodes = m_pNodes->getKernelData(devicePointer).Data;
	q.m_sStartNode = startNode;
	q.m_pNodeTransforms = m_pTransforms->getKernelData(devicePointer).Data;
	q.m_pInvNodeTransforms = m_pInvTransforms->getKernelData(devicePointer).Data;
	return q;
}

unsigned int e_SceneBVH::getSizeInBytes()
{
	return m_pNodes->getSizeInBytes();
}

const float4x4& e_SceneBVH::getNodeTransform(e_BufferReference<e_Node, e_Node> n)
{
	return *m_pTransforms->operator()(n.getIndex());
}

e_BVHNodeData* e_SceneBVH::getBVHNode(unsigned int i)
{
	if ((int)i >= m_sBvhNodeCount)
		return 0;
	e_StreamReference(e_BVHNodeData) n = m_pNodes->operator()(i);
	e_BVHNodeData* n2 = n.operator e_BVHNodeData *();
	return n2;
}

void e_SceneBVH::addNode(e_BufferReference<e_Node, e_Node> n)
{
	nodesToInsert.insert(n);
}

void e_SceneBVH::removeNode(e_BufferReference<e_Node, e_Node> n)
{
	nodesToRemove.insert(n);
	std::set<e_StreamReference(e_Node)>::iterator it = nodesToInsert.find(n);
	if (it != nodesToInsert.end())
		nodesToInsert.erase(it);
}

bool e_SceneBVH::needsBuild()
{
	return nodesToRecompute.size() != 0 || nodesToInsert.size() != 0 || nodesToRemove.size() != 0;
}

AABB e_SceneBVH::getWorldNodeBox(e_BufferReference<e_Node, e_Node> n)
{
	unsigned int mi = n->m_uMeshIndex;
	AABB box = info->a_Meshes(mi)->m_sLocalBox;
	float4x4 mat = *m_pTransforms[0](n.getIndex());
	return box.Transform(mat);
}

int e_SceneBVH::getChildIdxInLocal(BVHIndex nodeIdx, BVHIndex childIdx)
{
	BVHIndexTuple c = FromChildren(getBVHNode(nodeIdx.innerIdx()));
	if (c[0] != childIdx && c[1] != childIdx)
		throw std::runtime_error("Invalid tree passed!");
	return c[1] == childIdx;
}

void e_SceneBVH::sahModified(BVHIndex nodeIdx, const AABB& box, float& leftSAH, float& rightSAH)
{
	AABB left, right;
	e_BVHNodeData* node = m_pNodes->operator()(nodeIdx.innerIdx());
	node->getBox(left, right);
	BVHIndexTuple c = FromChildren(node);
	float lA = left.Area(), rA = right.Area();
	left.Enlarge(box);
	right.Enlarge(box);
	float lAd = left.Area(), rAd = right.Area();
	int lN = c[0].isLeaf() ? 1 : bvhNodeData[c[0].innerIdx()].numLeafs;
	int rN = c[1].isLeaf() ? 1 : bvhNodeData[c[1].innerIdx()].numLeafs;
	leftSAH = lAd * (lN + 1) + rA * rN;
	rightSAH = lA * lN + rAd * (rN + 1);
}

int e_SceneBVH::validateTree(BVHIndex idx, BVHIndex parent)
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
		if (info.parent != parent)
			throw std::runtime_error(__FUNCTION__);
		BVHIndexTuple c = FromChildren(m_pNodes->operator()(idx.innerIdx()));
		int s = validateTree(c[0], idx) + validateTree(c[1], idx);
		if (s != info.numLeafs)
			throw std::runtime_error(__FUNCTION__);
		return s;
	}
}

AABB e_SceneBVH::getBox(BVHIndex idx)
{
	if (idx.isLeaf())
		return getWorldNodeBox(info->a_Nodes(idx.leafIdx()));
	else return m_pNodes->operator()(idx.innerIdx())->getBox();
}

void e_SceneBVH::propagateFlag(BVHIndex idx)
{
	if (idx.isLeaf())
	{
		flaggedSceneNodes[idx.leafIdx()] = 1;
		propagateFlag(nodeToBVHNode[idx.leafIdx()]);
	}
	else
	{
		flaggedBVHNodes[idx.innerIdx()] = 1;
		BVHNodeInfo& info = bvhNodeData[idx.innerIdx()];
		if (info.parent.isValid())
			propagateFlag(info.parent);
	}
}

int e_SceneBVH::numberGrandchildren(BVHIndex idx, int localChildIdx)
{
	if (idx.isLeaf())
		throw std::runtime_error(__FUNCTION__);
	BVHIndexTuple c = FromChildren(m_pNodes->operator()(idx.innerIdx()));
	BVHIndex child = c[localChildIdx];
	if (child.isLeaf())
		return 0;
	BVHIndexTuple c2 = FromChildren(m_pNodes->operator()(child.innerIdx()));
	return (!c2[0].NoNode()) + (!c2[1].NoNode());
}

float e_SceneBVH::sah(BVHIndex lhs, BVHIndex rhs)
{
	AABB lBox = getBox(lhs), rBox = getBox(rhs);
	int lNum = lhs.isLeaf() ? 1 : bvhNodeData[lhs.innerIdx()].numLeafs;
	int rNum = rhs.isLeaf() ? 1 : bvhNodeData[rhs.innerIdx()].numLeafs;
	return lBox.Area() * lNum + rBox.Area() * rNum;
}

float e_SceneBVH::sah(BVHIndex idx, int localChildIdx, int localGrandChildIdx)
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

e_SceneBVH::BVHIndexTuple e_SceneBVH::children(BVHIndex idx)
{
	return FromChildren(m_pNodes->operator()(idx.innerIdx()));
}

int e_SceneBVH::numLeafs(BVHIndex idx)
{
	return idx.isLeaf() ? 1 : bvhNodeData[idx.innerIdx()].numLeafs;
}