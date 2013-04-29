#pragma once

#include "..\Math\vector.h"
#include "..\Math\AABB.h"

struct e_PointBVHNode
{
	AABB leftBox;
	AABB rightBox;
	int leftIndex;
	int rightIndex;
	void setData(int _leftStart, int _rightStart, AABB& lBox, AABB& rBox)
	{
		leftBox = lBox;
		leftIndex = _leftStart;
		rightBox = rBox;
		rightIndex = _rightStart;
	}
};

template<typename T> int BuildPointBVH(T* data, int a_Count, e_PointBVHNode* a_NodeOut, int* a_IndexOut, float maxRadius);