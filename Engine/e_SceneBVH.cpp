#include "StdAfx.h"
#include "e_SceneBVH.h"
#include <xmmintrin.h>

#define AREA(x) (2.0f * (x.m128_f32[3] * x.m128_f32[2] + x.m128_f32[2] * x.m128_f32[1] + x.m128_f32[3] * x.m128_f32[1]))
#define TOVEC3(x) make_float3(x.m128_f32[3], x.m128_f32[2], x.m128_f32[1])
#define TOSSE3(v) _mm_set_ps(v.x, v.y, v.z, 0)
#define TOBOX(b,t) AABB(TOVEC3(b), TOVEC3(t))
__declspec(align(128)) struct BBoxTmp
{
    __declspec(align(16)) __m128 _bottom;
    __declspec(align(16)) __m128 _top;
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
			float testSplit = n % 2 ? work[n / 2]._top.m128_f32[dim] : work[n / 2]._bottom.m128_f32[dim];
			__m128 lbottom(_mm_set1_ps(FLT_MAX)), ltop(_mm_set1_ps(-FLT_MAX));
			__m128 rbottom(_mm_set1_ps(FLT_MAX)), rtop(_mm_set1_ps(-FLT_MAX));
			for(int i = 0; i < n / 2; i++)
			{
				const BBoxTmp& v = work[i];
				lbottom = _mm_min_ps(lbottom, v._bottom);
				ltop = _mm_max_ps(ltop, v._top);
			}
			for(int i = n / 2; i < size; i++)
			{
				const BBoxTmp& v = work[i];
				rbottom = _mm_min_ps(rbottom, v._bottom);
				rtop = _mm_max_ps(rtop, v._top);
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

void e_SceneBVH::Build(e_Node* a_Nodes, unsigned int a_Count)
{
	__m128 bottom(_mm_set1_ps(FLT_MAX)), top(_mm_set1_ps(-FLT_MAX));
	BBoxTmp* data = (BBoxTmp*)_mm_malloc(a_Count * sizeof(BBoxTmp), 128);
	for(unsigned int i = 0; i < a_Count; i++)
	{
		AABB box = a_Nodes[i].getWorldBox();
		__m128 q0 = TOSSE3(box.minV), q1 = TOSSE3(box.maxV);
		data[i]._bottom = q0;
		data[i]._top = q1;
		data[i]._center = _mm_mul_ps(_mm_add_ps(data[i]._bottom, data[i]._top), _mm_set_ps(0.5f,0.5f,0.5f,1));
		data[i]._pNode = i;
		bottom = _mm_min_ps(bottom, data[i]._bottom);
		top = _mm_max_ps(top, data[i]._top);
		*m_pTransforms->getHost(i) = a_Nodes[i].getWorldMatrix().Transpose();
		*m_pInvTransforms->getHost(i) = a_Nodes[i].getInvWorldMatrix().Transpose();
	}
	if(a_Count)
	{
		startNode = RecurseNative(a_Count, data, nativelist<e_BVHNodeData>(m_pNodes->getHost(0)), bottom, top);
	}
	else
	{
		startNode = 0;
		m_pNodes->getHost(0)->setDummy();
		//m_sBox = a_Nodes->getWorldBox();
	}
	m_pNodes->Invalidate(DataStreamRefresh_Immediate);
	m_pTransforms->Invalidate(DataStreamRefresh_Immediate);
	m_pInvTransforms->Invalidate(DataStreamRefresh_Immediate);
	_mm_free(data);
	m_sBox = TOBOX(bottom, top);
}