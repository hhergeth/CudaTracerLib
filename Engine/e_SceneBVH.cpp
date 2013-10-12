#include "StdAfx.h"
#include "e_SceneBVH.h"
#include "e_Node.h"
#include <xmmintrin.h>
#include <algorithm>

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

#define MaxDepth 64
#define MAX_OBJECT_COUNT 1024 * 1024 * 5
struct buffer
{
private:
	struct entry
	{
		BBoxTmp* item;
		int index;
	};
	entry* skipBuffer[3];
	int* indices;
public:
	int N;
private:
	buffer(){}
public:
	buffer(int n, BBoxTmp* work)
	{
		struct cmp
		{
			int dim;
			cmp(int i) : dim(i){}
			bool operator()(entry& l, entry& r) const
			{
				BBoxTmp* left = l.item, *right = r.item;
				float ca = left->box.b.m128_f32[dim] + left->box.t.m128_f32[dim];
				float cb = right->box.b.m128_f32[dim] + right->box.t.m128_f32[dim];
				return (ca < cb || (ca == cb && left->_pNode < right->_pNode));
			}
		};
		N = n;
		for(int i = 0; i < 3; i++)
		{
			skipBuffer[i] = new entry[n];
			for(int j = 0; j < n; j++)
			{
				skipBuffer[i][j].index = j;
				skipBuffer[i][j].item = work + j;
			}
			std::make_heap(skipBuffer[i], skipBuffer[i] + n, cmp(i));
			std::sort_heap(skipBuffer[i], skipBuffer[i] + n, cmp(i));
		}
		indices = new int[n * 3];
		for(int i = 0; i < N; i++)
		{
			//we are iterating over SLOTS not objects
			for(int j = 0; j < 3; j++)
				indices[skipBuffer[j][i].index * 3 + j] = i;
		}
	}

	void Free()
	{
		delete indices;
		for(int i = 0; i < 3; i++)
			delete [] skipBuffer[i];
	}

	bool validate()
	{
		for(int i = 0; i < N; i++)
		{
			for(int j = 0; j < 3; j++)
				if(indices[i * 3 + j] != skipBuffer[j][i].index)
					throw 1;
		}
	}

	buffer part(int dim, int start, int n)
	{
		validate();
		buffer b;
		b.N = n;
		b.indices = new int[n * 3];
		for(int pseudoDim = 0; pseudoDim < 3; pseudoDim++)
		{
			int arr[] = {dim,(dim+1)%3,(dim+2)%3};
			int i = arr[pseudoDim];
			b.skipBuffer[i] = new entry[n];
			int c = 0, end = start + n;
			for(int j = 0; j < N; j++)//iterate over all elements
			{
				//determine whether the current should be inserted
				int indexInSortedDim = indices[j * 3 + i];
				if(indexInSortedDim >= start && indexInSortedDim < end)
				{
					b.skipBuffer[i][c].item = skipBuffer[dim][j].item;
					int index = skipBuffer[dim][j].index - start;
					b.skipBuffer[i][c].index = index;
					b.indices[index * 3 + i] = c;
					c++;
				}
			}
			if(c != n)
				throw 1;
		}
		b.validate();
		return b;
	}
	BBoxTmp* operator()(int dim, int i)
	{

	}
};

static __m128_box* m_rightBounds = new __m128_box[MAX_OBJECT_COUNT]; 
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
			leftBounds.Enlarge(buf(m_sortDim, i)->box);
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
static int* m_indices = new int[MAX_OBJECT_COUNT];
static int m_indicesCounter = 0;
int createLeaf(buffer& buf)
{
	int start = ~m_indicesCounter;
	for(int i = 0; i < buf.N; i++)
		m_indices[i] = buf(0, i)->_pNode;
	m_indicesCounter += buf.N;
	return start;
}
int performObjectSplit(buffer& buf, nativelist<e_BVHNodeData>& a_Nodes, __m128_box& box, Platform& P, int level=0)
{
	if (buf.N <= P.getMinLeafSize() || level >= MaxDepth)
		return createLeaf(buf);
	float area = box.area();
	float leafSAH = area * P.getTriangleCost(buf.N);
    float nodeSAH = area * P.getNodeCost(2);
    ObjectSplit object = findObjectSplit(buf, P, nodeSAH);
    if (nodeSAH == leafSAH && buf.N <= P.getMaxLeafSize())
        return createLeaf(buf);
	buffer leftB = buf.part(object.sortDim, 0, object.numLeft);
	buffer rightB = buf.part(object.sortDim, object.numLeft, buf.N - object.numLeft);
	e_BVHNodeData* n = a_Nodes.Add();
	n->setLeft(object.leftBounds.ToBox());
	n->setRight(object.rightBounds.ToBox());
	int ld = performObjectSplit(leftB, a_Nodes, object.leftBounds, P, level + 1);
	int rd = performObjectSplit(rightB, a_Nodes, object.rightBounds, P, level + 1);
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
		//startNode = RecurseNative(a_Nodes.getLength(), data, nativelist<e_BVHNodeData>(nds.operator->()), bottom, top);
		startNode = performObjectSplit(buffer(a_Nodes.getLength(), data), nativelist<e_BVHNodeData>(nds.operator->()), __m128_box(bottom, top), Platform(1));
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