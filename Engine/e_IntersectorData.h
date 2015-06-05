#pragma once

#include "../Math/Vector.h"
#include "../Math/AABB.h"

struct Ray;

#define MAX_AREALIGHT_NUM 2

struct e_TriIntersectorData2
{
	unsigned int index;
public:
	CUDA_FUNC_IN void setIndex(unsigned int i)
	{
		index = (i << 1) | (index & 1);
	}
	CUDA_FUNC_IN void setFlag(bool b)
	{
		index = (index & ~1) | !!b;
	}
	CUDA_FUNC_IN unsigned int getIndex()
	{
		return index >> 1;
	}
	CUDA_FUNC_IN bool getFlag()
	{
		return index & 1;
	}
};

struct e_TriIntersectorData
{
private:
	Vec4f a, b, c;
public:
	CUDA_DEVICE CUDA_HOST void setData(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2);

	CUDA_DEVICE CUDA_HOST void getData(Vec3f& v0, Vec3f& v1, Vec3f& v2) const;

	CUDA_DEVICE CUDA_HOST bool Intersect(const Ray& r, float* dist = 0, Vec2f* bary = 0) const;
};

struct e_BVHNodeData
{
	//      nodes[innerOfs + 0 ] = Vec4f(c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
	//      nodes[innerOfs + 16] = Vec4f(c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
	//      nodes[innerOfs + 32] = Vec4f(c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
	Vec4f a, b, c, d;
	CUDA_FUNC_IN const Vec2i& getChildren() const
	{
		return *(Vec2i*)&d;
	}
	CUDA_FUNC_IN Vec2i& getChildren()
	{
		return *(Vec2i*)&d;
	}
	CUDA_FUNC_IN void getBox(AABB& left, AABB& right) const
	{
		left.minV = Vec3f(a.x, a.z, c.x);
		left.maxV = Vec3f(a.y, a.w, c.y);
		right.minV = Vec3f(b.x, b.z, c.z);
		right.maxV = Vec3f(b.y, b.w, c.w);
	}
	CUDA_FUNC_IN AABB getLeft() const
	{
		return AABB(Vec3f(a.x, a.z, c.x), Vec3f(a.y, a.w, c.y));
	}
	CUDA_FUNC_IN AABB getRight() const
	{
		return AABB(Vec3f(b.x, b.z, c.z), Vec3f(b.y, b.w, c.w));
	}
	CUDA_FUNC_IN void setBox(const AABB& c0, const AABB& c1)
	{
		a = Vec4f(c0.minV.x, c0.maxV.x, c0.minV.y, c0.maxV.y);
		b = Vec4f(c1.minV.x, c1.maxV.x, c1.minV.y, c1.maxV.y);
		c = Vec4f(c0.minV.z, c0.maxV.z, c1.minV.z, c1.maxV.z);
	}
	CUDA_FUNC_IN void setLeft(const AABB& c0)
	{
		a = Vec4f(c0.minV.x, c0.maxV.x, c0.minV.y, c0.maxV.y);
		c.x = c0.minV.z;
		c.y = c0.maxV.z;
	}
	CUDA_FUNC_IN void setRight(const AABB& c1)
	{
		b = Vec4f(c1.minV.x, c1.maxV.x, c1.minV.y, c1.maxV.y);
		c.z = c1.minV.z;
		c.w = c1.maxV.z;
	}
	CUDA_FUNC_IN void setChildren(const Vec2i& c)
	{
		*(Vec2i*)&d = c;
	}
	CUDA_FUNC_IN void setParent(unsigned int parent)
	{
		*(unsigned int*)&d.z = parent;
	}
	CUDA_FUNC_IN void setSibling(unsigned int sibling)
	{
		*(unsigned int*)&d.w = sibling;
	}
	CUDA_FUNC_IN void setDummy()
	{
		AABB std(Vec3f(0), Vec3f(0));
		setLeft(std);
		setRight(std);
		setChildren(Vec2i(0, 0));
	}

	CUDA_FUNC_IN AABB getBox()
	{
		AABB left, right;
		getBox(left, right);
		left.Enlarge(right);
		return left;
	}

	CUDA_FUNC_IN void setChild(int localChildIdx, int childIdx, const AABB& box)
	{
		getChildren()[localChildIdx] = childIdx;
		if (localChildIdx == 0)
			setLeft(box);
		else setRight(box);
	}
};