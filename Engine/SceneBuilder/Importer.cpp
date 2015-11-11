#include <StdAfx.h>
#include <Base/FileStream.h>
#include "Importer.h"
#include "SplitBVHBuilder.hpp"

namespace CudaTracerLib {

namespace bvh_helper
{
class clb : public IBVHBuilderCallback
{
	const Vec3f* V;
	const unsigned int* Iab;
	unsigned int v, i;
	unsigned int _index(unsigned int i, unsigned int o) const
	{
		return Iab ? Iab[i * 3 + o] : (i * 3 + o);
	}
public:
	std::vector<e_BVHNodeData>& nodes;
	std::vector<e_TriIntersectorData>& tris;
	std::vector<e_TriIntersectorData2>& indices;
	AABB box;
	int startNode;
	//unsigned int L0, L1;
	unsigned int l0, l1;
public:
	clb(unsigned int _v, unsigned int _i, const Vec3f* _V, const unsigned int* _I, std::vector<e_BVHNodeData>& A, std::vector<e_TriIntersectorData>& B, std::vector<e_TriIntersectorData2>& C)
		: V(_V), Iab(_I), v(_v), i(_i), nodes(A), tris(B), indices(C), l0(0), l1(0)
	{
	}
	virtual void setNumInner_Leaf(unsigned int nInnerNodes, unsigned int nLeafNodes)
	{
		nodes.resize(nInnerNodes + 2);
		tris.resize(nLeafNodes + 2);
		indices.resize(nLeafNodes + 2);
	}
	virtual void iterateObjects(std::function<void(unsigned int)> f)
	{
		for (unsigned int j = 0; j < i / 3; j++)
			f(j);
	}
	virtual void getBox(unsigned int index, AABB* out) const
	{
		*out = AABB::Identity();
		for (int i = 0; i < 3; i++)
			*out = out->Extend(V[_index(index, i)]);
		//if(min(out->Size()) < 0.01f)
		//	out->maxV += make_float3(0.01f);
	}
	virtual void HandleBoundingBox(const AABB& box)
	{
		this->box = box;
	}
	virtual e_BVHNodeData* HandleNodeAllocation(int* index)
	{
		size_t n = l0++;
		if (n >= nodes.size())
			throw std::runtime_error(__FUNCTION__);
		*index = (int)n * 4;
		return &nodes[n];
	}
	void setSibling(int idx, int sibling)
	{
		if (idx >= 0)
			nodes[idx / 4].setSibling(sibling);
		else
		{
			int o = ~idx;
			while (!indices[o].getFlag())
				o++;
			o += 2;
			//*(int*)(indices + o) = sibling;
		}
	}
	virtual unsigned int handleLeafObjects(unsigned int pNode)
	{
		size_t c = l1++;
		if (c >= indices.size())
			throw std::runtime_error(__FUNCTION__);
		tris[c].setData(V[_index(pNode, 0)], V[_index(pNode, 1)], V[_index(pNode, 2)]);
		indices[c].setFlag(false);
		indices[c].setIndex(pNode);
		return (unsigned int)c;
	}
	virtual void handleLastLeafObject(int parent)
	{
		indices[l1 - 1].setFlag(true);
	}
	virtual void HandleStartNode(int startNode)
	{
		this->startNode = startNode;
	}
	virtual bool SplitNode(unsigned int index, int dim, float pos, AABB& lBox, AABB& rBox, const AABB& refBox) const
	{
		lBox = rBox = AABB::Identity();
		Vec3f v1 = V[_index(index, 2)];
		for (int i = 0; i < 3; i++)
		{
			Vec3f v0 = v1;
			v1 = V[_index(index, i)];
			float V0[] = { v0.x, v0.y, v0.z };
			float V1[] = { v1.x, v1.y, v1.z };
			float v0p = V0[dim];
			float v1p = V1[dim];

			// Insert vertex to the boxes it belongs to.

			if (v0p <= pos)
				lBox = lBox.Extend(v0);
			if (v0p >= pos)
				rBox = rBox.Extend(v0);

			// Edge intersects the plane => insert intersection to both boxes.

			if ((v0p < pos && v1p > pos) || (v0p > pos && v1p < pos))
			{
				Vec3f t = math::lerp(v0, v1, math::clamp01((pos - v0p) / (v1p - v0p)));
				lBox = lBox.Extend(t);
				rBox = rBox.Extend(t);
			}
		}
		lBox.maxV[dim] = pos;
		rBox.minV[dim] = pos;
		lBox = lBox.Intersect(refBox);
		rBox = rBox.Intersect(refBox);
		return true;
	}
};
}

void ConstructBVH(const Vec3f* vertices, const unsigned int* indices, unsigned int vCount, unsigned int cCount, BVH_Construction_Result& out)
{
	bvh_helper::clb c(vCount, cCount, vertices, indices, out.nodes, out.tris, out.tris2);
	SplitBVHBuilder::Platform P; P.m_maxLeafSize = 8;
	SplitBVHBuilder bu(&c, P, SplitBVHBuilder::BuildParams());
	bu.run();
	BVH_Construction_Result r;
	r.box = c.box;
	r.tris2 = c.indices;
	r.nodes = c.nodes;
	r.tris = c.tris;
}

void ConstructBVH(const Vec3f* vertices, const unsigned int* indices, int vCount, int cCount, FileOutputStream& O, BVH_Construction_Result* out)
{
	BVH_Construction_Result localRes;
	if (!out)
		out = &localRes;

	bvh_helper::clb c(vCount, cCount, vertices, indices, out->nodes, out->tris, out->tris2);
	SplitBVHBuilder::Platform P; P.m_maxLeafSize = 8;
	SplitBVHBuilder bu(&c, P, SplitBVHBuilder::BuildParams()); bu.run();
	O << (unsigned long long)c.l0;
	if (c.l0)
		O.Write(&c.nodes[0], (unsigned int)c.l0 * sizeof(e_BVHNodeData));
	O << (unsigned long long)c.l1;
	if (c.l1)
		O.Write(&c.tris[0], (unsigned int)c.l1 * sizeof(e_TriIntersectorData));
	O << (unsigned long long)c.l1;
	if (c.l1)
		O.Write(&c.indices[0], (unsigned int)c.l1 * sizeof(e_TriIntersectorData2));
}

}