#include "StdAfx.h"
#include "e_SceneBVH.h"
#include "e_Node.h"
#include <xmmintrin.h>
#include <algorithm>
#include "..\Base\Timer.h"

#define TOVEC3(x) make_float3(x.m128_f32[2], x.m128_f32[1], x.m128_f32[0])
#define TOSSE3(v) _mm_set_ps(0, v.x, v.y, v.z)
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
};

#define AREA(x) (2.0f * (x.m128_f32[2] * x.m128_f32[1] + x.m128_f32[1] * x.m128_f32[0] + x.m128_f32[2] * x.m128_f32[0]))
#define TOBOX(b,t) AABB(TOVEC3(b), TOVEC3(t))

__declspec(align(128)) struct BBoxTmp
{
    __m128_box box;
    __declspec(align(16)) __m128 _center;
	unsigned int _pNode;
};

template<typename T> class nativelist
{
public:
	T* buffer;
	unsigned int p;
	nativelist(T* a)
	{
		buffer = a;
		p = 0;
	}
	T* Add()
	{
		return &buffer[p++];
	}
	void Add(T& a)
	{
		buffer[p++] = a;
	}
	unsigned int index(T* a)
	{
		return ((unsigned long long)a - (unsigned long long)buffer) / sizeof(T);
	}
	inline T* operator[](int n) { return &buffer[n]; }
};

struct ObjectSplit
{
    float                 sah;
    int                 sortDim;
    int                 numLeft;
    __m128_box                leftBounds;
    __m128_box                rightBounds;

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
    float		            sah;
    int			            dim;
    float	                pos;

    SpatialSplit(void) : sah(FLT_MAX), dim(0), pos(0.0f) {}
};

class Platform
{
public:
    Platform(int maxLeaf)
	{ 
		m_SAHNodeCost = 1.f;
		m_SAHTriangleCost = 1.f;
		m_nodeBatchSize = 1;
		m_triBatchSize = 1;
		m_minLeafSize=1;
		m_maxLeafSize=maxLeaf;
	}
    Platform(float nodeCost=1.f, float triCost=1.f, int nodeBatchSize=1, int triBatchSize=1) {  m_SAHNodeCost = nodeCost; m_SAHTriangleCost = triCost; m_nodeBatchSize = nodeBatchSize; m_triBatchSize = triBatchSize; m_minLeafSize=1; m_maxLeafSize=0x7FFFFFF; }

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
private:
    float   m_SAHNodeCost;
    float   m_SAHTriangleCost;
    int     m_triBatchSize;
    int     m_nodeBatchSize;
    int     m_minLeafSize;
    int     m_maxLeafSize;
};

#define MaxSpatialDepth 48
#define MaxDepth 64
#define MAX_OBJECT_COUNT 1024 * 1024 * 5
#define NumSpatialBins 128
static __m128 binScale = _mm_set_ps1(1.0f / float(NumSpatialBins)), psZero = _mm_set_ps1(0), psBinClamp = _mm_set_ps1(NumSpatialBins - 1);
struct buffer
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
	buffer(){}
	int initialN;
	std::vector<BBoxTmp> refAdds;//we have to keep the additional values
public:
	buffer(int n, BBoxTmp* work)
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
		initialN = N = n;
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
			entries[sortedBuffers[0][i]].item = work[i];
			//we are iterating over SLOTS not objects
			for(int j = 0; j < 3; j++)
				entries[sortedBuffers[j][i]].indices[j] = i;
		}
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

	buffer part(int dim, int start, int n)
	{
		buffer b;
		b.N = n;
		b.entries = new entry[n];
		//b.initialN = n;
		//b.initialValues = 
		for(int i = 0; i < 3; i++)
		{
			b.sortedBuffers[i] = new int[n];
			int c = 0, end = start + n;
			for(int j = 0; j < N; j++)//iterate over all elements
			{
				//determine whether the current should be inserted
				int indexInSortedDim = entries[j].indices[dim];
				if(indexInSortedDim >= start && indexInSortedDim < end)
				{
					//okay so we should insert this object
					//new index in current dimension will be c!

					//but we also need the index of the object...
					int index = indexInSortedDim - start;
					b.sortedBuffers[i][c] = index;
					b.entries[index].indices[i] = c;
					b.entries[index].item = entries[j].item;
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
			buffer* b;
			cmp(int i,buffer* a) : dim(i),b(a){}
			bool operator()(int l, int r) const
			{
				BBoxTmp& left = l < b->initialN ? *b->operator[](l) : b->refAdds[l - b->initialN], 
					   & right = r < b->initialN ? *b->operator[](l) : b->refAdds[r - b->initialN];
				float ca = left.box.b.m128_f32[dim] + left.box.t.m128_f32[dim];
				float cb = right.box.b.m128_f32[dim] + right.box.t.m128_f32[dim];
				return (ca < cb || (ca == cb && left._pNode < right._pNode));
			}
		};
		refAdds.insert(refAdds.end(), refs.begin(), refs.end());
		Free();
		N += refs.size();
		entries = new entry[N];
		for(int i = 0; i < initialN; i++)
			entries[i].item = *operator[](i);
		for(int j = 0; j < refAdds.size(); j++)
			entries[initialN + j].item = refAdds[j];
		for(int i = 0; i < 3; i++)
		{
			sortedBuffers[i] = new int[N];
			for(int j = 0; j < initialN; j++)
				sortedBuffers[i][j] = j;
			int c = initialN;
			for(int j = 0; j < refAdds.size(); j++)
				sortedBuffers[i][c] = c++;
			std::make_heap(sortedBuffers[i], sortedBuffers[i] + N, cmp(i, this));
			std::sort_heap(sortedBuffers[i], sortedBuffers[i] + N, cmp(i, this));
		}
		for(int i = 0; i < N; i++)
			for(int j = 0; j < 3; j++)
				entries[sortedBuffers[j][i]].indices[j] = i;
		validate();
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

static __m128_box* m_rightBounds = new __m128_box[MAX_OBJECT_COUNT];
static SpatialBin* m_bins[3] = {new SpatialBin[NumSpatialBins],new SpatialBin[NumSpatialBins],new SpatialBin[NumSpatialBins]};
ObjectSplit findObjectSplit(buffer& buf, Platform& P, float nodeSAH)
{
	int numRef = buf.N;
	ObjectSplit split;
	float bestTieBreak = FLT_MAX;
	for (int m_sortDim = 0; m_sortDim < 3; m_sortDim++)
	{
		__m128_box rightBounds = __m128_box::Identity();
        for (int i = numRef - 1; i > 0; i--)
        {
			rightBounds.Enlarge(buf(m_sortDim, i)->box);
            m_rightBounds[i - 1] = rightBounds;
        }
		__m128_box leftBounds = __m128_box::Identity();
        for (int i = 1; i < numRef; i++)
        {
			leftBounds.Enlarge(buf(m_sortDim, i - 1)->box);
            float sah = nodeSAH + leftBounds.area() * P.getTriangleCost(i) + m_rightBounds[i - 1].area() * P.getTriangleCost(numRef - i);
            float tieBreak = sqrtf((float)i) + sqrtf((float)(numRef - i));
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
__m128 clampToBin(__m128& v)
{
	return _mm_min_ps (_mm_max_ps(v, psZero), psBinClamp);
}
void splitReference(BBoxTmp& left, BBoxTmp& right, const BBoxTmp* ref, int dim, float pos)
{
	left._pNode = right._pNode = ref->_pNode;
    left.box = right.box = AABB();
	if(ref->box.b.m128_f32[dim] < pos && pos < ref->box.t.m128_f32[dim])
	{

	}
	left.box.t.m128_f32[dim] = pos;
	right.box.b.m128_f32[dim] = pos;
	left.box.Intersect(ref->box);
	right.box.Intersect(ref->box);
}
SpatialSplit findSpatialSplit(buffer& buf, __m128_box& box, Platform& P, float nodeSAH)
{
	__m128 origin = box.b;
	__m128 binSize = _mm_mul_ps(_mm_sub_ps(box.t, origin), binScale);
    __m128 invBinSize = _mm_mul_ps(_mm_set_ps1(1.0f), binSize);

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
		__m128 firstBin = clampToBin(_mm_mul_ps(_mm_sub_ps(buf(0, refIdx)->box.b, origin), invBinSize));
		__m128 lastBin = clampToBin(_mm_mul_ps(_mm_sub_ps(buf(0, refIdx)->box.t, origin), invBinSize));

        for (int dim = 0; dim < 3; dim++)
        {
            BBoxTmp currRef = *buf(0, refIdx);
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
struct NodeSpec
{
        int                 numRef;
		__m128_box          bounds;

        NodeSpec(void) : numRef(0) {}
};
int createLeaf(buffer& buf)
{
	return ~buf(0, 0)->_pNode;
}
void performObjectSplit(buffer& buf, NodeSpec& left, NodeSpec& right, ObjectSplit& split, Platform& P)
{
    left.numRef = split.numLeft;
    left.bounds = split.leftBounds;
	right.numRef = buf.N - split.numLeft;
    right.bounds = split.rightBounds;
}
void performSpatialSplit(buffer& buf, NodeSpec& left, NodeSpec& right, SpatialSplit& split, Platform& P)
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
int buildNode(buffer& buf, nativelist<e_BVHNodeData>& a_Nodes, __m128_box& box, Platform& P, float m_minOverlap, int level=0)
{
	if (buf.N <= P.getMinLeafSize() || level >= MaxDepth)
		return createLeaf(buf);
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
            spatial = findSpatialSplit(buf, box, P, nodeSAH);
    }
	float minSAH = MIN(leafSAH, object.sah, spatial.sah);
	if (minSAH == leafSAH && buf.N <= P.getMaxLeafSize())
        return createLeaf(buf);
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
	e_BVHNodeData* n = a_Nodes.Add();
	n->setLeft(left.bounds.ToBox());
	n->setRight(right.bounds.ToBox());
	int ld = buildNode(leftB, a_Nodes, left.bounds, P, level + 1);
	int rd = buildNode(rightB, a_Nodes, right.bounds, P, level + 1);
	n->setChildren(make_int2(ld, rd));
	leftB.Free();
	rightB.Free();
	return a_Nodes.index(n);
}

int RecurseNative(int size, BBoxTmp* work, nativelist<e_BVHNodeData>& a_Nodes, __m128& ssebottom, __m128& ssetop, int depth=0)
{
    if (size == 1)
		return ~work[0]._pNode;
	__m128 a0 = _mm_sub_ps(ssetop, ssebottom);
	float area0 = AREA(a0);
	int bestCountLeft, bestCountRight;
	__m128 bestLeftBottom, bestLeftTop, bestRightBottom, bestRightTop;
	float bestCost = FLT_MAX;
    float bestSplit = FLT_MAX;
	float bestTieBreak = FLT_MAX;
	int bestDim = -1;
	for(int dim = 1; dim < 4; dim++)
	{
		for(int n = 0; n < size * 2; n++)
		{
			float testSplit = n % 2 ? work[n / 2].box.t.m128_f32[dim] : work[n / 2].box.b.m128_f32[dim];
			__m128 lbottom(_mm_set1_ps(FLT_MAX)), ltop(_mm_set1_ps(-FLT_MAX));
			__m128 rbottom(_mm_set1_ps(FLT_MAX)), rtop(_mm_set1_ps(-FLT_MAX));
			for(int i = 0; i < n / 2; i++)
			{
				const BBoxTmp& v = work[i];
				lbottom = _mm_min_ps(lbottom, v.box.b);
				ltop = _mm_max_ps(ltop, v.box.t);
			}
			for(int i = n / 2; i < size; i++)
			{
				const BBoxTmp& v = work[i];
				rbottom = _mm_min_ps(rbottom, v.box.b);
				rtop = _mm_max_ps(rtop, v.box.t);
			}
			int countLeft = n / 2, countRight = size - n / 2;
			if (countLeft<1 || countRight<1)
				continue;

			__m128 ltopMinusBottom = _mm_sub_ps(ltop, lbottom);
			__m128 rtopMinusBottom = _mm_sub_ps(rtop, rbottom);
			float totalCost = .125f + (countLeft * AREA(ltopMinusBottom) + countRight * AREA(rtopMinusBottom)) / area0;
			float tieBreak = sqrtf((float)n / 2) + sqrtf((float)(size - n / 2));
			if (totalCost < bestCost || (totalCost == bestCost && tieBreak < bestTieBreak))
			{
				bestTieBreak = tieBreak;
				bestDim = dim;
				bestCost = totalCost;
				bestSplit = testSplit;
				bestCountLeft = countLeft;
				bestCountRight = countRight;
				bestLeftBottom = lbottom;
				bestLeftTop = ltop;
				bestRightBottom = rbottom;
				bestRightTop = rtop;
			}
		}
	}
	if(bestDim == -1)
		throw 1;
	BBoxTmp* left = (BBoxTmp*)_mm_malloc(bestCountLeft * sizeof(BBoxTmp), 128), *right = (BBoxTmp*)_mm_malloc(bestCountRight * sizeof(BBoxTmp), 128);
	int l = bestCountLeft, r = bestCountRight;
	for(int i = 0; i < bestCountLeft; i++)
		left[i] = work[i];
	for(int i = 0; i < bestCountRight; i++)
		right[i] = work[bestCountLeft + i];
	e_BVHNodeData* n = a_Nodes.Add();
	int ld = RecurseNative(l, left, a_Nodes, bestLeftBottom, bestLeftTop, depth + 1),
		rd = RecurseNative(r, right, a_Nodes, bestRightBottom, bestRightTop, depth + 1);
	n->setLeft(TOBOX(bestLeftBottom, bestLeftTop));
	n->setRight(TOBOX(bestRightBottom, bestRightTop));
	n->setChildren(make_int2(ld, rd));
	_mm_free(left);
	_mm_free(right);
	return a_Nodes.index(n);
}

void e_SceneBVH::Build(e_StreamReference(e_Node) a_Nodes, e_BufferReference<e_Mesh, e_KernelMesh> a_Meshes)
{
	__m128 bottom(_mm_set1_ps(FLT_MAX)), top(_mm_set1_ps(-FLT_MAX));
	BBoxTmp* data = (BBoxTmp*)_mm_malloc(a_Nodes.getLength() * sizeof(BBoxTmp), 128);
	for(unsigned int i = 0; i < a_Nodes.getLength(); i++)
	{
		AABB box = a_Nodes(i)->getWorldBox(a_Meshes(a_Nodes(i)->m_uMeshIndex));
		__m128 q0 = TOSSE3(box.minV), q1 = TOSSE3(box.maxV);
		data[i].box.b = q0;
		data[i].box.t = q1;
		data[i]._center = _mm_mul_ps(_mm_add_ps(data[i].box.b, data[i].box.t), _mm_set_ps(0.5f,0.5f,0.5f,1));
		data[i]._pNode = i;
		bottom = _mm_min_ps(bottom, data[i].box.b);
		top = _mm_max_ps(top, data[i].box.t);
		m_pTransforms->operator()(i) = a_Nodes[i].getWorldMatrix().Transpose();
		m_pInvTransforms->operator()(i) = a_Nodes[i].getInvWorldMatrix().Transpose();
	}
	if(a_Nodes.getLength())
	{
		cTimer T;
		T.StartTimer();
		//startNode = RecurseNative(a_Nodes.getLength(), data, nativelist<e_BVHNodeData>(nds.operator->()), bottom, top);
		Platform P(1);
		float mo = __m128_box(bottom, top).area() * 1.0e-5f;
		startNode = buildNode(buffer(a_Nodes.getLength(), data), nativelist<e_BVHNodeData>(nds.operator->()),  __m128_box(bottom, top), P, mo);
		double tSec = T.EndTimer();
		std::cout << "BVH Construction of " << a_Nodes.getLength() << " objects took " << tSec << " seconds\n";
	}
	else
	{
		startNode = 0;
		m_pNodes->operator()(0)->setDummy();
		//m_sBox = a_Nodes->getWorldBox();
	}
	m_pNodes->Invalidate();
	m_pNodes->UpdateInvalidated();
	m_pTransforms->Invalidate();
	m_pTransforms->UpdateInvalidated();
	m_pInvTransforms->Invalidate();
	m_pInvTransforms->UpdateInvalidated();
	_mm_free(data);
	m_sBox = TOBOX(bottom, top);
}

e_SceneBVH::e_SceneBVH(unsigned int a_NodeCount)
{
	m_pNodes = new e_Stream<e_BVHNodeData>(a_NodeCount * 2);//largest binary tree has the same amount of inner nodes
	m_pTransforms = new e_Stream<float4x4>(a_NodeCount);
	m_pInvTransforms = new e_Stream<float4x4>(a_NodeCount);
	startNode = -1;
	m_sBox = AABB::Identity();
	tr0 = m_pTransforms->malloc(m_pTransforms->getLength());
	tr1 = m_pInvTransforms->malloc(m_pInvTransforms->getLength());
	nds = m_pNodes->malloc(m_pNodes->getLength());
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