#include <StdAfx.h>
#include "..\..\Base\FileStream.h"
#include "Importer.h"
#include "SplitBVHBuilder.hpp"

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
		e_BVHNodeData* nodes;
		unsigned int nodeIndex;
		e_TriIntersectorData* tris;
		unsigned int triIndex;
		e_TriIntersectorData2* indices;
		AABB box;
		int startNode;
		unsigned int L0, L1;
	public:
		clb(unsigned int _v, unsigned int _i, const Vec3f* _V, const unsigned int* _I)
			: V(_V), Iab(_I), v(_v), i(_i)
		{
			nodeIndex = triIndex = 0;
			const float duplicates = 0.5f, f = 1.0f + duplicates;
			L0 = (int)(float(_i / 3) * f);
			L1 = (int)(float(_i / 3) * f * 4);
			nodes = new e_BVHNodeData[L0];
			tris = new e_TriIntersectorData[L1];
			indices = new e_TriIntersectorData2[L1];
			Platform::SetMemory(indices, L1 * sizeof(e_TriIntersectorData2));
		}
		void Free()
		{
			delete[] nodes;
			delete[] tris;
			delete[] indices;
		}
		virtual unsigned int Count() const
		{
			return i / 3;
		}
		virtual void getBox(unsigned int index, AABB* out) const
		{
			*out = AABB::Identity();
			for (int i = 0; i < 3; i++)
				out->Enlarge(V[_index(index, i)]);
			//if(min(out->Size()) < 0.01f)
			//	out->maxV += make_float3(0.01f);
		}
		virtual void HandleBoundingBox(const AABB& box)
		{
			this->box = box;
		}
		virtual e_BVHNodeData* HandleNodeAllocation(int* index)
		{
			*index = nodeIndex++ * 4;
			if (nodeIndex > L0)
				throw std::runtime_error("Not enough space for nodes!");
			return nodes + *index / 4;
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
			unsigned int c = triIndex++;
			if (triIndex > L1)
				throw std::runtime_error("Not enough space for leafes!");
			indices[c].setIndex(pNode);
			tris[c].setData(V[_index(pNode, 0)], V[_index(pNode, 1)], V[_index(pNode, 2)]);
			indices[c].setFlag(false);
			indices[c].setIndex(pNode);
			return c;
		}
		virtual void handleLastLeafObject(int parent)
		{
			//*(int*)&tris[triIndex].x = 0x80000000;
			//indices[triIndex] = -1;
			//triIndex += 1;
			//if(triIndex > L1)
			//	throw 1;
			indices[triIndex - 1].setFlag(true);
			//*(int*)(indices + triIndex) = parent;
			//triIndex += 2;
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
					lBox.Enlarge(v0);
				if (v0p >= pos)
					rBox.Enlarge(v0);

				// Edge intersects the plane => insert intersection to both boxes.

				if ((v0p < pos && v1p > pos) || (v0p > pos && v1p < pos))
				{
					Vec3f t = math::lerp(v0, v1, math::clamp01((pos - v0p) / (v1p - v0p)));
					lBox.Enlarge(t);
					rBox.Enlarge(t);
				}
			}
			lBox.maxV[dim] = pos;
			rBox.minV[dim] = pos;
			lBox.intersect(refBox);
			rBox.intersect(refBox);
			return true;
		}
	};
}

BVH_Construction_Result ConstructBVH(const Vec3f* vertices, const unsigned int* indices, unsigned int vCount, unsigned int cCount)
{
	bvh_helper::clb c(vCount, cCount, vertices, indices);
	//BVHBuilder::BuildBVH(&c, BVHBuilder::Platform());
	SplitBVHBuilder bu(&c, SplitBVHBuilder::Platform(), SplitBVHBuilder::BuildParams()); bu.run();
	BVH_Construction_Result r;
	r.box = c.box;
	r.tris2 = c.indices;
	r.nodeCount = c.nodeIndex;
	r.nodes = c.nodes;
	r.triCount = c.triIndex;
	r.tris = c.tris;
	return r;
}

void ConstructBVH(const Vec3f* vertices, const unsigned int* indices, int vCount, int cCount, OutputStream& O, BVH_Construction_Result* out)
{	
	bvh_helper::clb c(vCount, cCount, vertices, indices);
	//BVHBuilder::BuildBVH(&c, BVHBuilder::Platform());
	SplitBVHBuilder bu(&c, SplitBVHBuilder::Platform(), SplitBVHBuilder::BuildParams()); bu.run();
	O << (unsigned long long)c.nodeIndex;
	if(c.nodeIndex)
		O.Write(c.nodes, (unsigned int)c.nodeIndex * sizeof(e_BVHNodeData));
	O << (unsigned long long)c.triIndex;
	if(c.triIndex )
		O.Write(c.tris, (unsigned int)c.triIndex * sizeof(e_TriIntersectorData));
	O << (unsigned long long)c.triIndex;
	if(c.triIndex )
		O.Write(c.indices, (unsigned int)c.triIndex * sizeof(e_TriIntersectorData2));
	if(out)
	{
		out->box = c.box;
		out->tris2 = c.indices;
		out->nodeCount = c.nodeIndex;
		out->nodes = c.nodes;
		out->triCount = c.triIndex;
		out->tris = c.tris;
	}
	else c.Free();
}