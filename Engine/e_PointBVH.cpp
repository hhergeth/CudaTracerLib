#include "StdAfx.h"
#include "e_PointBVH.h"
#include <xmmintrin.h>

#define AREA(x) (2.0f * (x.m128_f32[3] * x.m128_f32[2] + x.m128_f32[2] * x.m128_f32[1] + x.m128_f32[3] * x.m128_f32[1]))
#define TOVEC3(x) make_float3(x.m128_f32[3], x.m128_f32[2], x.m128_f32[1])
#define TOSSE3(v) _mm_set_ps(v.x, v.y, v.z, 0)
#define TOBOX(b,t) AABB(TOVEC3(b), TOVEC3(t))
#define V_P_R0(min, max, i) (max.m128_f32[i] - min.m128_f32[i] + 2.0f * maxRadius)
#define V_P_R(min, max) (V_P_R0(min, max, 1) * V_P_R0(min, max, 2) * V_P_R0(min, max, 3))

__declspec(align(128)) struct BBoxTmp
{
    __declspec(align(16)) __m128 _pos;
	unsigned int _index;
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
		return ((unsigned int)a - (unsigned int)buffer) / sizeof(T);
	}
	inline T* operator[](int n) { return &buffer[n]; }
};

int addleaf(BBoxTmp* lwork, int lsize, nativelist<unsigned int>& a_Indices, nativelist<e_PointBVHNode>& a_Nodes)
{
	int start = a_Indices.p;
	for(int j=0; j<lsize; j++)
		a_Indices.Add(lwork[j]._index);
	unsigned int end = -1;
	a_Indices.Add(end);
	return ~start;
}

int RecurseNative(float maxRadius, int size, BBoxTmp* work, nativelist<unsigned int>& a_Indices, nativelist<e_PointBVHNode>& a_Nodes, __m128& ssebottom, __m128& ssetop, int depth=0)
{
	const int minCount = 4;
    if (size <= minCount)
		return addleaf(work, size, a_Indices, a_Nodes);
	__m128 a0 = _mm_sub_ps(ssetop, ssebottom);
	float area0 = AREA(a0);
	int bestCountLeft, bestCountRight;
	__m128 bestLeftBottom, bestLeftTop, bestRightBottom, bestRightTop;
	float bestCost = FLT_MAX;
	int bestDim = -1;
	for(int dim = 1; dim < 4; dim++)
	{
		for(int n = 0; n < size; n++)
		{
			__m128 lbottom(_mm_set1_ps(FLT_MAX)), ltop(_mm_set1_ps(-FLT_MAX));
			__m128 rbottom(_mm_set1_ps(FLT_MAX)), rtop(_mm_set1_ps(-FLT_MAX));
			for(int i = 0; i < n / 2; i++)
				lbottom = _mm_min_ps(lbottom, work[i]._pos);
			for(int i = n / 2; i < size; i++)
				rbottom = _mm_min_ps(rbottom, work[i]._pos);
			int countLeft = n / 2, countRight = size - n / 2;
			if (countLeft<1 || countRight<1)
				continue;
			__m128 ltopMinusBottom = _mm_sub_ps(ltop, lbottom);
			__m128 rtopMinusBottom = _mm_sub_ps(rtop, rbottom);
			float vt = V_P_R(ssebottom, ssetop), totalCost = 1.0f + V_P_R(lbottom, ltop) / vt * (float)countLeft + V_P_R(rbottom, rtop) / vt * (float)countRight;
			if (totalCost < bestCost)
			{
				bestDim = dim;
				bestCost = totalCost;
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
		return addleaf(work, size, a_Indices, a_Nodes);
	BBoxTmp* left = (BBoxTmp*)_mm_malloc(bestCountLeft * sizeof(BBoxTmp), 128), *right = (BBoxTmp*)_mm_malloc(bestCountRight * sizeof(BBoxTmp), 128);
	int l = bestCountLeft, r = bestCountRight;
	for(int i = 0; i < bestCountLeft; i++)
		left[i] = work[i];
	for(int i = 0; i < bestCountRight; i++)
		right[i] = work[bestCountLeft + i];
	e_PointBVHNode* n = a_Nodes.Add();
	int ld = RecurseNative(maxRadius, l, left, a_Indices, a_Nodes, bestLeftBottom, bestLeftTop, depth + 1),
		rd = RecurseNative(maxRadius, r, right, a_Indices, a_Nodes, bestRightBottom, bestRightTop, depth + 1); 
	n->setData(ld, rd, TOBOX(bestLeftBottom, bestLeftTop), TOBOX(bestRightBottom, bestRightTop));
	_mm_free(left);
	_mm_free(right);
	return a_Nodes.index(n);
}

template<typename T> int BuildPointBVH(T* data, int a_Count, e_PointBVHNode* a_NodeOut, int* a_IndexOut, float maxRadius)
{
	__m128 bottom(_mm_set1_ps(FLT_MAX)), top(_mm_set1_ps(-FLT_MAX));
	BBoxTmp* data = (BBoxTmp*)_mm_malloc(a_Count * sizeof(BBoxTmp), 128);
	for(unsigned int i = 0; i < a_Count; i++)
	{
		if(!data[i].isValid())
			continue;
		data[i]._center = data[i].getPos();
		data[i]._pNode = i;
		bottom = _mm_min_ps(bottom, data[i]._center);
		top = _mm_max_ps(top, data[i]._center);
	}
	startNode = RecurseNative(maxRadius, a_Count, data, nativelist<unsigned int>(m_pIndices->getHost(0)), nativelist<e_PointBVHNode>(m_pNodes->getHost(0)), bottom, top);
	_mm_free(data);
}