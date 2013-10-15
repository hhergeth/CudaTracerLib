#include <StdAfx.h>
#include "BVHBuilder.h"
#include <xmmintrin.h>
#include <vector>
#include <algorithm>
#include "..\Base\Timer.h"
#include <iostream>
void print(const __m128& v)
{
	std::cout << "{" << v.m128_f32[0] << ", " << v.m128_f32[1] << ", " << v.m128_f32[2] << ", " << v.m128_f32[3] << "}\n";
}

#define TOVEC3(x) make_float3(x.m128_f32[0], x.m128_f32[1], x.m128_f32[2])
#define TOSSE3(v) _mm_set_ps(0, v.z, v.y, v.x)
struct __m128_box
{
	__m128 b;
	__m128 t;
	__m128_box() {}
	__m128_box(const __m128& min, const __m128& max)
		: b(min), t(max)
	{
	}
	static __m128_box Identity()
	{
		return __m128_box(AABB::Identity());
	}
	__m128_box(const AABB& box)
	{
		b = TOSSE3(box.minV);
		t = TOSSE3(box.maxV);
	}
	float area()
	{
		if(!isValid())
			return 0.0f;
		__m128 x = _mm_sub_ps(t, b);
		return 2.0f * (x.m128_f32[2] * x.m128_f32[1] + x.m128_f32[1] * x.m128_f32[0] + x.m128_f32[2] * x.m128_f32[0]);
	}
	AABB ToBox()
	{
		return AABB(TOVEC3(b), TOVEC3(t));
	}
	void Enlarge(const __m128_box& box)
	{
		b = _mm_min_ps(b, box.b);
		t = _mm_max_ps(t, box.t);
	}
	void Intersect(const __m128_box& aabb)
	{
		b = _mm_max_ps(b, aabb.b);
		t = _mm_min_ps(t, aabb.t);
	}
	bool isValid()
	{
		return b.m128_f32[0] <= t.m128_f32[0] && b.m128_f32[1] <= t.m128_f32[1] && b.m128_f32[2] <= t.m128_f32[2];
	}
};

struct BBoxTmp
{
    __m128_box box;
	unsigned int _pNode;
};

struct ObjectSplit
{
    float               sah;
    int                 sortDim;
    int                 numLeft;
    __m128_box          leftBounds;
    __m128_box          rightBounds;

    ObjectSplit(void) : sah(FLT_MAX), sortDim(0), numLeft(0) {}
};

struct SpatialBin
{
    __m128_box          bounds;
    int                 enter;
    int                 exit;
};

struct SpatialSplit
{
    float		        sah;
    int			        dim;
    float				pos;

    SpatialSplit(void) : sah(FLT_MAX), dim(0), pos(0.0f) {}
};

struct NodeSpec
{
        int                 numRef;
		__m128_box          bounds;

        NodeSpec(void) : numRef(0) {}
};

#define MaxSpatialDepth 48
#define MaxDepth 64
#define MAX_OBJECT_COUNT 1024 * 1024 * 12
#define NumSpatialBins 128
static __m128 binScale = _mm_set_ps1(1.0f / float(NumSpatialBins)), psZero = _mm_set_ps1(0), psBinClamp = _mm_set_ps1(NumSpatialBins - 1);
struct buffer2
{
private:
	struct entry
	{
		BBoxTmp item;
		int indices[3];//indices into sortedBuffers
	};
	int* sortedBuffers[3];//indices into entries
	entry* entries;
public:
	int N;
private:
	buffer2(){}
public:
	buffer2(int n, BBoxTmp* work)
	{
		struct cmp
		{
			int dim;
			BBoxTmp* work;
			cmp(int i,BBoxTmp* a) : dim(i),work(a){}
			bool operator()(int l, int r) const
			{
				BBoxTmp& left = work[l], &right = work[r];
				float ca = left.box.b.m128_f32[dim] + left.box.t.m128_f32[dim];
				float cb = right.box.b.m128_f32[dim] + right.box.t.m128_f32[dim];
				return (ca < cb || (ca == cb && left._pNode < right._pNode));
			}
		};
		N = n;
		for(int i = 0; i < 3; i++)
		{
			sortedBuffers[i] = new int[n];
			for(int j = 0; j < n; j++)
				sortedBuffers[i][j] = j;
			std::make_heap(sortedBuffers[i], sortedBuffers[i] + n, cmp(i, work));
			std::sort_heap(sortedBuffers[i], sortedBuffers[i] + n, cmp(i, work));
		}
		entries = new entry[n];
		for(int i = 0; i < N; i++)
		{
			entries[sortedBuffers[0][i]].item = work[sortedBuffers[0][i]];
			//we are iterating over SLOTS not objects
			for(int j = 0; j < 3; j++)
				entries[sortedBuffers[j][i]].indices[j] = i;
		}
		delete [] work;
	}

	void Free()
	{
		delete entries;
		for(int i = 0; i < 3; i++)
			delete [] sortedBuffers[i];
	}

	bool validate()
	{
		//return true;
		for(int i = 0; i < N; i++)
		{
			for(int j = 0; j < 3; j++)
				if(sortedBuffers[j][entries[i].indices[j]] != i)
					throw 1;
		}
	}

	buffer2 part(int dim, int start, int n)
	{
		buffer2 b;
		b.N = n;
		b.entries = new entry[n];
		for(int i = 0; i < 3; i++)
		{
			b.sortedBuffers[i] = new int[n];
			int c = 0, end = start + n;
			for(int indexInIteratingDim = 0; indexInIteratingDim < N; indexInIteratingDim++)//iterate over all elements
			{
				//determine whether the current should be inserted
				int entryIndex = sortedBuffers[i][indexInIteratingDim];
				int indexInSortedDim = entries[entryIndex].indices[dim];
				if(indexInSortedDim >= start && indexInSortedDim < end)
				{
					//okay so we should insert this object
					//new index in current dimension will be c!

					//but we also need the index of the object...
					int index = indexInSortedDim - start;//start is the beginning in the sorted dim
					b.sortedBuffers[i][c] = index;
					b.entries[index].indices[i] = c;
					b.entries[index].item = entries[entryIndex].item;
					c++;
				}
			}
			if(c != n)
				throw 1;
		}
		return b;
	}

	void appendAndRebuild(std::vector<BBoxTmp>& refs)
	{
		struct cmp
		{
			int dim;
			buffer2* b;
			cmp(int i,buffer2* a) : dim(i),b(a){}
			bool operator()(int l, int r) const
			{
				BBoxTmp& left = *b->operator[](l), 
					   & right = *b->operator[](r);
				float ca = left.box.b.m128_f32[dim] + left.box.t.m128_f32[dim];
				float cb = right.box.b.m128_f32[dim] + right.box.t.m128_f32[dim];
				return (ca < cb || (ca == cb && left._pNode < right._pNode));
			}
		};
		entry* newEntries = new entry[N + refs.size()];
		for(int i = 0; i < N; i++)
			newEntries[i].item = *operator[](i);
		for(int j = 0; j < refs.size(); j++)
			newEntries[N + j].item = refs[j];
		Free();
		entries = newEntries;
		N += (int)refs.size();
		for(int i = 0; i < 3; i++)
		{
			sortedBuffers[i] = new int[N];
			for(int j = 0; j < N; j++)
				sortedBuffers[i][j] = j;
			std::make_heap(sortedBuffers[i], sortedBuffers[i] + N, cmp(i, this));
			std::sort_heap(sortedBuffers[i], sortedBuffers[i] + N, cmp(i, this));
		}
		for(int i = 0; i < N; i++)
			for(int j = 0; j < 3; j++)
				entries[sortedBuffers[j][i]].indices[j] = i;
		validate();
	}

	void sort(int dim)
	{
	}

	BBoxTmp* operator()(int dim, int i)
	{
		return &entries[sortedBuffers[dim][i]].item;
	}
	BBoxTmp* operator[](int i)
	{
		return operator()(0, i);
	}
};
#define buffer buffer2
struct buffer3
{
private:
	BBoxTmp* items;
	int sortedDim;
	buffer3(){}
public:
	int N;
public:
	buffer3(int n, BBoxTmp* work)
	{
		items = work;
		sortedDim = -1;
		N = n;
	}
	void Free()
	{
		delete [] items;
	}
	buffer3 part(int dim, int start, int n)
	{
		sort(dim);
		buffer3 r;
		r.N = n;
		r.sortedDim = -1;
		r.items = new BBoxTmp[n];
		memcpy(r.items, items + start, n * sizeof(BBoxTmp));
		return r;
	}
	void appendAndRebuild(std::vector<BBoxTmp>& refs)
	{
		if(refs.size() == 0)
			return;
		BBoxTmp* a = new BBoxTmp[N + refs.size()];
		memcpy(a, items, sizeof(BBoxTmp) * N);
		memcpy(a + N, &refs[0], sizeof(BBoxTmp) * refs.size());
		free(items);
		items = a;
		N += refs.size();
		int d = sortedDim;
		sortedDim = -1;
		sort(d);
	}
	void sort(int dim)
	{
		if(dim == sortedDim)
			return;
		sortedDim = dim;
		struct cmp
		{
			int dim;
			cmp(int i) : dim(i){}
			bool operator()(BBoxTmp& left, BBoxTmp& right) const
			{
				float ca = left.box.b.m128_f32[dim] + left.box.t.m128_f32[dim];
				float cb = right.box.b.m128_f32[dim] + right.box.t.m128_f32[dim];
				return (ca < cb || (ca == cb && left._pNode < right._pNode));
			}
		};
		std::make_heap(items, items + N, cmp(sortedDim));
		std::sort_heap(items, items + N, cmp(sortedDim));
	}
	void removeDegenerates()
	{
		for(int i = N - 1; i >= 0; i--)
		{
			__m128 s = _mm_sub_ps(items[0].box.t, items[0].box.b);
			float mi = MIN(s.m128_f32[0], s.m128_f32[1], s.m128_f32[2]), ma = MAX(s.m128_f32[0], s.m128_f32[1], s.m128_f32[2]);
			if(mi < 0.0f || (s.m128_f32[0] + s.m128_f32[1] + s.m128_f32[2]) == ma)
			{
				BBoxTmp old = items[i];
				N--;
				if(i < N)
					items[i] = items[N];
			}
		}
	}
	BBoxTmp* operator()(int dim, int i)
	{
		if(dim != sortedDim)
			throw 1;
		return items + i;
	}
	BBoxTmp* operator[](int i)
	{
		return items + i;
	}
};

float sqr(float f){return f*f;}
static __m128_box* m_rightBounds = new __m128_box[MAX_OBJECT_COUNT];
static SpatialBin* m_bins[3] = {new SpatialBin[NumSpatialBins],new SpatialBin[NumSpatialBins],new SpatialBin[NumSpatialBins]};
ObjectSplit findObjectSplit(buffer& buf, const BVHBuilder::Platform& P, float nodeSAH)
{
	int numRef = buf.N;
	ObjectSplit split;
	if(numRef == 1)
	{
		split.sah = 0.0f;
		split.numLeft = 1;
		split.sortDim = 0;
		split.leftBounds = buf[0]->box;
		split.rightBounds = __m128_box(AABB(make_float3(0), make_float3(0)));
		return split;
	}
	float bestTieBreak = FLT_MAX;
	for (int m_sortDim = 0; m_sortDim < 3; m_sortDim++)
	{
		buf.sort(m_sortDim);
		__m128_box rightBounds = __m128_box::Identity();
        for (int i = numRef - 1; i > 0; i--)
        {
			rightBounds.Enlarge(buf(m_sortDim, i)->box);
            m_rightBounds[i - 1] = rightBounds;
        }
		__m128_box leftBounds = __m128_box::Identity();
        for (int i = 1; i < numRef; i++)
        {
			__m128_box bb = buf(m_sortDim, i - 1)->box;
			leftBounds.Enlarge(bb);
			float lA = leftBounds.area(), rA = m_rightBounds[i - 1].area();
            float sah = nodeSAH + lA * P.getTriangleCost(i) + rA * P.getTriangleCost(numRef - i);
            float tieBreak = sqr((float)i) + sqr((float)(numRef - i));
            if (sah < split.sah || (sah == split.sah && tieBreak < bestTieBreak))
            {
                split.sah = sah;
                split.sortDim = m_sortDim;
                split.numLeft = i;
                split.leftBounds = leftBounds;
                split.rightBounds = m_rightBounds[i - 1];
                bestTieBreak = tieBreak;
            }
        }
	}
	return split;
}
void splitReference(BBoxTmp& left, BBoxTmp& right, const BBoxTmp* ref, int dim, float pos)
{
	left._pNode = right._pNode = ref->_pNode;
    left.box = right.box = __m128_box::Identity();
	if(ref->box.b.m128_f32[dim] < pos && pos < ref->box.t.m128_f32[dim])
	{
		left.box = right.box = ref->box;
	}/*
	left.box.t.m128_f32[dim] = pos;
	right.box.b.m128_f32[dim] = pos;
	left.box.Intersect(ref->box);
	right.box.Intersect(ref->box);*/
	//no idea what whould be the correct way to do it
	//but can one actually split a AABB further without supplementary information?
}
SpatialSplit findSpatialSplit(buffer& buf, __m128_box& box, const BVHBuilder::Platform& P, float nodeSAH)
{
	__m128 origin = box.b;
	__m128 binSize = _mm_mul_ps(_mm_sub_ps(box.t, origin), binScale);
    __m128 invBinSize = _mm_div_ps(_mm_set_ps1(1.0f), binSize);

    for (int dim = 0; dim < 3; dim++)
    {
        for (int i = 0; i < NumSpatialBins; i++)
        {
            SpatialBin& bin = m_bins[dim][i];
			bin.bounds = __m128_box::Identity();
            bin.enter = 0;
            bin.exit = 0;
        }
    }

	for (int refIdx = 0; refIdx < buf.N; refIdx++)
    {
		__m128 b = buf[refIdx]->box.b, t = buf[refIdx]->box.t;
		__m128 firstBin = _mm_mul_ps(_mm_sub_ps(b, origin), invBinSize);
		__m128 lastBin = _mm_mul_ps(_mm_sub_ps(t, origin), invBinSize);
		firstBin = _mm_min_ps (_mm_max_ps(firstBin, psZero), psBinClamp);
		lastBin = _mm_min_ps (_mm_max_ps(lastBin, firstBin), psBinClamp);

        for (int dim = 0; dim < 3; dim++)
        {
            BBoxTmp currRef = *buf[refIdx];
			for (int i = firstBin.m128_f32[dim]; i < lastBin.m128_f32[dim]; i++)
            {
                BBoxTmp leftRef, rightRef;
				splitReference(leftRef, rightRef, &currRef, dim, origin.m128_f32[dim] + binSize.m128_f32[dim] * (float)(i + 1));
				m_bins[dim][i].bounds.Enlarge(leftRef.box);
                currRef = rightRef;
            }
			m_bins[dim][(int)lastBin.m128_f32[dim]].bounds.Enlarge(currRef.box);
            m_bins[dim][(int)firstBin.m128_f32[dim]].enter++;
            m_bins[dim][(int)lastBin.m128_f32[dim]].exit++;
        }
    }

    SpatialSplit split;
    for (int dim = 0; dim < 3; dim++)
    {
        // Sweep right to left and determine bounds.

		__m128_box rightBounds = __m128_box::Identity();
        for (int i = NumSpatialBins - 1; i > 0; i--)
        {
			rightBounds.Enlarge(m_bins[dim][i].bounds);
            m_rightBounds[i - 1] = rightBounds;
        }

        // Sweep left to right and select lowest SAH.

        __m128_box leftBounds = __m128_box::Identity();
        int leftNum = 0;
		int rightNum = buf.N;

        for (int i = 1; i < NumSpatialBins; i++)
        {
            leftBounds.Enlarge(m_bins[dim][i - 1].bounds);
            leftNum += m_bins[dim][i - 1].enter;
            rightNum -= m_bins[dim][i - 1].exit;

            float sah = nodeSAH + leftBounds.area() * P.getTriangleCost(leftNum) + m_rightBounds[i - 1].area() * P.getTriangleCost(rightNum);
            if (sah < split.sah)
            {
                split.sah = sah;
                split.dim = dim;
				split.pos = origin.m128_f32[dim] + binSize.m128_f32[dim] * (float)i;
            }
        }
    }
    return split;
}
int createLeaf(buffer& buf, IBVHBuilderCallback* clb)
{
	if(!buf.N)
		return -214783648;
	unsigned int start = clb->handleLeafObjects(buf[0]->_pNode);
	for(int i = 1; i < buf.N; i++)
		clb->handleLeafObjects(buf[i]->_pNode);
	clb->handleLastLeafObject();
	return ~start;
}
void performObjectSplit(buffer& buf, NodeSpec& left, NodeSpec& right, ObjectSplit& split, const BVHBuilder::Platform& P)
{
    left.numRef = split.numLeft;
    left.bounds = split.leftBounds;
	right.numRef = buf.N - split.numLeft;
    right.bounds = split.rightBounds;
}
void performSpatialSplit(buffer& buf, NodeSpec& left, NodeSpec& right, SpatialSplit& split, const BVHBuilder::Platform& P)
{
	int leftStart = 0;
    int leftEnd = leftStart;
	int rightStart = buf.N;
	left.bounds = __m128_box::Identity();
	right.bounds = __m128_box::Identity();
	for (int i = leftEnd; i < rightStart; i++)
    {
        // Entirely on the left-hand side?

		if (buf[i]->box.t.m128_f32[split.dim] <= split.pos)
        {
			left.bounds.Enlarge(buf[i]->box);
            swapk(buf[i], buf[leftEnd++]);
        }

        // Entirely on the right-hand side?

		else if (buf[i]->box.b.m128_f32[split.dim] >= split.pos)
        {
			right.bounds.Enlarge(buf[i]->box);
            swapk(buf[i--], buf[--rightStart]);
        }
    }

	std::vector<BBoxTmp> refs;
	refs.reserve(MAX(buf.N / 10, 128));
	while (leftEnd < rightStart)
    {
        // Split reference.

        BBoxTmp lref, rref;
        splitReference(lref, rref, buf[leftEnd], split.dim, split.pos);

        // Compute SAH for duplicate/unsplit candidates.

        __m128_box lub = left.bounds;  // Unsplit to left:     new left-hand bounds.
        __m128_box rub = right.bounds; // Unsplit to right:    new right-hand bounds.
        __m128_box ldb = left.bounds;  // Duplicate:           new left-hand bounds.
        __m128_box rdb = right.bounds; // Duplicate:           new right-hand bounds.
        lub.Enlarge(buf[leftEnd]->box);
        rub.Enlarge(buf[leftEnd]->box);
        ldb.Enlarge(lref.box);
        rdb.Enlarge(rref.box);

        float lac = P.getTriangleCost(leftEnd - leftStart);
		float rac = P.getTriangleCost(buf.N - rightStart);
        float lbc = P.getTriangleCost(leftEnd - leftStart + 1);
        float rbc = P.getTriangleCost(buf.N - rightStart + 1);
		
        float unsplitLeftSAH = lub.area() * lbc + right.bounds.area() * rac;
		float unsplitRightSAH = left.bounds.area() * lac + rub.area() * rbc;
        float duplicateSAH = ldb.area() * lbc + rdb.area() * rbc;
        float minSAH = MIN(unsplitLeftSAH, unsplitRightSAH, duplicateSAH);

        // Unsplit to left?

        if (minSAH == unsplitLeftSAH)
        {
            left.bounds = lub;
            leftEnd++;
        }

        // Unsplit to right?

        else if (minSAH == unsplitRightSAH)
        {
            right.bounds = rub;
            swapk(buf[leftEnd], buf[--rightStart]);
        }

        // Duplicate?

        else
        {
            left.bounds = ldb;
            right.bounds = rdb;
            *buf[leftEnd++] = lref;
			refs.push_back(rref);
        }
    }
	buf.appendAndRebuild(refs);

    left.numRef = leftEnd - leftStart;
	right.numRef = buf.N - rightStart;
}
int buildNode(buffer& buf,  __m128_box& box, const BVHBuilder::Platform& P, float m_minOverlap, IBVHBuilderCallback* clb, int level=0)
{
	//buf.removeDegenerates();
	if ((buf.N <= P.getMinLeafSize() && level > 0) || level >= MaxDepth)
		return createLeaf(buf, clb);
	float area = box.area();
	float leafSAH = area * P.getTriangleCost(buf.N);
    float nodeSAH = area * P.getNodeCost(2);
    ObjectSplit object = findObjectSplit(buf, P, nodeSAH);
	SpatialSplit spatial;
    if (level < MaxSpatialDepth)
    {
        __m128_box overlap = object.leftBounds;
        overlap.Intersect(object.rightBounds);
        if (overlap.area() >= m_minOverlap)
		{
            spatial = findSpatialSplit(buf, box, P, nodeSAH);
		}
    }
	float minSAH = MIN(leafSAH, object.sah, spatial.sah);
	if (minSAH == leafSAH && buf.N <= P.getMaxLeafSize() && level > 0)
        return createLeaf(buf, clb);
    NodeSpec left, right;
	int dim;
    if (minSAH == spatial.sah)
	{
        performSpatialSplit(buf, left, right, spatial, P);
		dim = spatial.dim;
	}
    if (!left.numRef || !right.numRef)
	{
        performObjectSplit(buf, left, right, object, P);
		dim = object.sortDim;
	}
	buffer leftB = buf.part(dim, 0, left.numRef);
	buffer rightB = buf.part(dim, left.numRef, right.numRef);
	int index;
	e_BVHNodeData* n = clb->HandleNodeAllocation(&index);
	n->setLeft(left.bounds.ToBox());
	n->setRight(right.bounds.ToBox());
	int rd = buildNode(rightB, right.bounds, P, m_minOverlap, clb, level + 1);
	int ld = buildNode(leftB, left.bounds, P, m_minOverlap, clb, level + 1);
	n->setChildren(make_int2(ld, rd));
	leftB.Free();
	rightB.Free();
	return index;
}

void BVHBuilder::BuildBVH(IBVHBuilderCallback* clb, const BVHBuilder::Platform& P)
{
	unsigned int N = clb->Count();
	__m128 bottom(_mm_set1_ps(FLT_MAX)), top(_mm_set1_ps(-FLT_MAX));
	BBoxTmp* data = new BBoxTmp[N];
	AABB box;
	for(unsigned int i = 0; i < N; i++)
	{
		clb->getBox(i, &box);
		//if(fmaxf(box.Size()) < 0.01f)
		//	box.maxV += make_float3(0.01f);
		data[i].box.b = TOSSE3(box.minV);
		data[i].box.t = TOSSE3(box.maxV);
		data[i]._pNode = i;
		bottom = _mm_min_ps(bottom, data[i].box.b);
		top = _mm_max_ps(top, data[i].box.t);
	}
	__m128_box sbox = __m128_box(bottom, top);
	clb->HandleBoundingBox(sbox.ToBox());
	if(N)
	{
		cTimer T;
		T.StartTimer();
		float mo = __m128_box(bottom, top).area() * 1.0e-5f;
		buffer B = buffer(N, data);
		int sNode = buildNode(B,  sbox, P, mo, clb);
		clb->HandleStartNode(sNode);
		B.Free();
		double tSec = T.EndTimer();
		std::cout << "BVH Construction of " << N << " objects took " << tSec << " seconds\n";
	}
	else
	{
		clb->HandleStartNode(0);
		int idx;
		clb->HandleNodeAllocation(&idx)->setDummy();
		_mm_free(data);
	}
}