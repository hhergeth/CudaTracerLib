#include "StdAfx.h"
#include "e_SceneBVH.h"
#include "e_Mesh.h"
#include "e_Node.h"
#include "SceneBuilder/SplitBVHBuilder.hpp"

#define NO_NODE 0x76543210

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

struct BVHIndexTuple
{
	e_SceneBVH::BVHIndex values[2];

	e_SceneBVH::BVHIndex& operator[](int i)
	{
		return values[i];
	}

	const e_SceneBVH::BVHIndex& operator[](int i) const
	{
		return values[i];
	}
};

BVHIndexTuple FromChildren(const e_BVHNodeData* n)
{
	BVHIndexTuple t;
	for (int i = 0; i < 2; i++)
		t.values[i] = e_SceneBVH::BVHIndex::FromNative(n->getChildren()[i]);
	return t;
}

Vec2i ToChildren(const BVHIndexTuple& t)
{
	return Vec2i(t[0].ToNative(), t[1].ToNative());
}

struct e_SceneBVH::BVHNodeInfo
{
	BVHIndex parent;
	int flags;
	int numLeafs;

	BVHNodeInfo()
		: flags(0), numLeafs(0)
	{

	}

	BVHNodeInfo(BVHIndex p, int l)
		: parent(p), flags(0), numLeafs(l)
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
	AABB box;
	if (childIdx.isLeaf())
	{
		box = getWorldNodeBox(info->a_Nodes(childIdx.leafIdx()));
		nodeToBVHNode[childIdx.leafIdx()] = nodeIdx;
	}
	else
	{
		box = m_pNodes->operator()(childIdx.innerIdx())->getBox();
		bvhNodeData[childIdx.innerIdx()].parent = nodeIdx;
	}
	m_pNodes->operator()(nodeIdx.innerIdx())->setChild(localIdxToSetTo, childIdx.ToNative(), box);
}

void e_SceneBVH::removeNodeAndCollapse(BVHIndex nodeIdx, BVHIndex childIdx)
{
	int childIdxLocal = getChildIdxInLocal(nodeIdx, childIdx);
	e_BVHNodeData* node = m_pNodes->operator()(nodeIdx.innerIdx());
	BVHIndexTuple children = FromChildren(node);
	node->setChild(childIdxLocal, NO_NODE, AABB::Identity());
	bvhNodeData[nodeIdx.innerIdx()].changeCount(bvhNodeData , - 1);
	BVHIndex grandpaIdx = bvhNodeData[nodeIdx.innerIdx()].parent;
	if (children[1 - childIdxLocal].NoNode() && grandpaIdx.isValid())
	{
		//remove this node and propagate?
		removeNodeAndCollapse(grandpaIdx, nodeIdx);
	}
	else if (grandpaIdx.isValid())//there has to be one inner node in the tree
	{
		int parentLocal = getChildIdxInLocal(grandpaIdx, nodeIdx);
		setChild(grandpaIdx, children[1 - childIdxLocal], parentLocal);
	}
}

void e_SceneBVH::insertNode(BVHIndex bvhNodeIdx, BVHIndex nodeIdx, const AABB& nodeWorldBox)
{
	/*if (bvhNodeIdx < 0)//split leaf
	{
		e_BVHNodeData* node = m_pNodes->operator()(m_sBvhNodeCount);
		setChild(m_sBvhNodeCount, nodeIdx, 0);
		setChild(m_sBvhNodeCount, bvhNodeIdx, 1);
		int parentIdx = nodeToBVHNode[bvhNodeIdx];
		bvhNodeData[m_sBvhNodeCount] = BVHNodeInfo(parentIdx, 1);//only set one so we can increment
		setChild(parentIdx, m_sBvhNodeCount, getChildIdxInLocal(parentIdx, bvhNodeIdx));
		bvhNodeData[m_sBvhNodeCount].changeCount(bvhNodeData, +1);
		nodeToBVHNode[nodeIdx] = m_sBvhNodeCount;
		nodeToBVHNode[~bvhNodeIdx] = m_sBvhNodeCount;
		m_sBvhNodeCount++;
	}
	else
	{
		e_BVHNodeData* node = m_pNodes->operator()(bvhNodeIdx);
		Vec2i& c = node->getChildren();
		if (c.x == NO_NODE || c.y == NO_NODE)//insert into one child
		{
			node->setChild(c.y == NO_NODE, TO_CHILD(nodeIdx), getWorldNodeBox(info->a_Nodes(nodeIdx)));
			bvhNodeData[bvhNodeIdx].increaseCount(bvhNodeData);
			nodeToBVHNode[nodeIdx] = bvhNodeIdx;
		}
		else
		{
			float left_sah, right_sah;
			sahModified(bvhNodeIdx, nodeWorldBox, left_sah, right_sah);
			insertNode(left_sah < right_sah ? c.x : c.y, nodeIdx, nodeWorldBox);
		}
	}*/
}

bool e_SceneBVH::Build(e_StreamReference(e_Node) a_Nodes, e_BufferReference<e_Mesh, e_KernelMesh> a_Meshes)
{
	SceneInfo lInfo(a_Nodes, a_Meshes);
	info = &lInfo;
	bool modified = false;
	if (a_Nodes.getLength() != 0 && needsBuild())
	{
		size_t m = max(nodesToInsert.size(), nodesToRecompute.size(), nodesToRemove.size());
		if (m > a_Nodes.getLength() / 2 || startNode == -1)
		{
			BuilderCLB b(this);
			SplitBVHBuilder::Platform Pq;
			Pq.m_maxLeafSize = 1;
			SplitBVHBuilder bu(&b, Pq, SplitBVHBuilder::BuildParams());
			bu.run();
			b.buildInfoTree(BVHIndex::FromBVHNode(m_pNodes->operator()(startNode)), INVALID);
			validateTree(BVHIndex::FromBVHNode(m_pNodes->operator()(startNode)), INVALID);
		}
		else
		{
			for (size_t i = 0; i < nodesToRemove.size(); i++)
				removeNodeAndCollapse(nodeToBVHNode[nodesToRemove[i].getIndex()], BVHIndex::FromSceneNode(nodesToRemove[i]));
			m_sBox = m_pNodes->operator()(0)->getBox();
			for (size_t i = 0; i < nodesToInsert.size(); i++)
				insertNode(BVHIndex::FromBVHNode(m_pNodes->operator()(startNode)), BVHIndex::FromSceneNode(nodesToRemove[i]), getWorldNodeBox(nodesToRemove[i]));
			m_sBox = m_pNodes->operator()(0)->getBox();

			validateTree(BVHIndex::FromBVHNode(m_pNodes->operator()(startNode)), INVALID);
		}
	}
	if (!a_Nodes.getLength())
		startNode = -1;
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
}

void e_SceneBVH::invalidateNode(e_BufferReference<e_Node, e_Node> n)
{
	nodesToRecompute.push_back(n);
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
	e_StreamReference(e_BVHNodeData) n = m_pNodes->operator()(i);
	e_BVHNodeData* n2 = n.operator e_BVHNodeData *();
	return n2;
}

void e_SceneBVH::addNode(e_BufferReference<e_Node, e_Node> n)
{
	nodesToInsert.push_back(n);
}

void e_SceneBVH::removeNode(e_BufferReference<e_Node, e_Node> n)
{
	nodesToRemove.push_back(n);
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
	float lA = left.Area(), rA = right.Area();
	left.Enlarge(box);
	right.Enlarge(box);
	float lAd = left.Area(), rAd = right.Area();
	int lN = bvhNodeData[node->getChildren().x].numLeafs;
	int rN = bvhNodeData[node->getChildren().y].numLeafs;
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