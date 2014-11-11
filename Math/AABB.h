#pragma once
#include "cutil_math.h"
#include "float4x4.h"
#include "Ray.h"

struct AABB
{
	union
	{
		struct
		{
			float min[3];
			float max[3];
		};
		struct
		{
			float3 minV;
			float3 maxV;
		};
	};
	CUDA_FUNC_IN AABB(const float _min[3], const float _max[3])
	{
		min[0] = _min[0]; 
		min[1] = _min[1]; 
		min[2] = _min[2]; 
		max[0] = _max[0]; 
		max[1] = _max[1]; 
		max[2] = _max[2];
	}
	CUDA_FUNC_IN AABB()
	{
	}
	CUDA_FUNC_IN AABB(const float3& min, const float3& max)
	{
		minV = min;
		maxV = max;
	}
	CUDA_FUNC_IN void Enlarge(const AABB& a)
	{
		min[0] = MIN(min[0],a.min[0]);
		max[0] = MAX(max[0],a.max[0]);
		min[1] = MIN(min[1],a.min[1]);
		max[1] = MAX(max[1],a.max[1]);
		min[2] = MIN(min[2],a.min[2]);
		max[2] = MAX(max[2],a.max[2]);
	}
	CUDA_FUNC_IN void Enlarge(const float3& v)
	{
		minV = fminf(v, minV);
		maxV = fmaxf(v, maxV);
	}
	CUDA_FUNC_IN void intersect(const AABB& box)
	{
		minV = fmaxf(box.minV, minV);
		maxV = fminf(box.maxV, maxV);
	}
	CUDA_FUNC_IN float Area() const
	{
		float3 a = (maxV - minV);
		return 2.0f * (a.x * a.y + a.x * a.z + a.y * a.z);
	}
	CUDA_FUNC_IN float volume() const
	{
		float3 a = (maxV - minV);
		return a.x * a.y * a.z;
	}
	CUDA_FUNC_IN float w() const { return max[0]-min[0]; }
	CUDA_FUNC_IN float h() const { return max[1]-min[1]; }
	CUDA_FUNC_IN float d() const { return max[2]-min[2]; }
	CUDA_FUNC_IN AABB Transform(const float4x4& mat) const
	{
		float3 d = maxV - minV;
#define A(x,y,z) make_float3(x,y,z) * d + minV
		float3 v[8] = {A(0,0,0), A(1,0,0), A(1,0,1), A(0,0,1),
					   A(0,1,0), A(1,1,0), A(1,1,1), A(0,1,1)};
		float3 mi = make_float3(FLT_MAX), ma = make_float3(-FLT_MAX);
		for(int i = 0; i < 8; i++)
		{
			float3 q = mat.TransformPoint(v[i]);
			mi = fminf(q, mi);
			ma = fmaxf(q, ma);
		}
		return AABB(mi, ma);
#undef A
	}
	//Ensures that every dim != zero
	CUDA_FUNC_IN AABB Inflate() const
	{
		AABB b;
		b.minV = minV;
		b.maxV = maxV;
		for(int i = 0; i < 3; i++)
			if(b.max[i] - b.min[i] == 0)
			{
				b.max[i] += (float)EPSILON;
				b.min[i] -= (float)EPSILON;
			}
		return b;
	}
	//Enlarges the box by the factor
	CUDA_FUNC_IN AABB Enlarge(const float f = 0.015f) const
	{
		float3 q = (maxV - minV) / 2.0f, m = (maxV + minV) / 2.0f;
		float e2 = 1.0f + f;
		AABB box;
		box.maxV = m + q * e2;
		box.minV = m - q * e2;
		return box;
	}
	CUDA_FUNC_IN bool Contains(const float3& p) const
	{
		return minV.x <= p.x && p.x <= maxV.x && minV.y <= p.y && p.y <= maxV.y && minV.z <= p.z && p.z <= maxV.z;
	}
	CUDA_FUNC_IN float3 Size() const
	{
		return maxV - minV;
	}
	CUDA_FUNC_IN float3 Center() const
	{
		return (maxV + minV) / 2.0f;
	}
	static CUDA_FUNC_IN AABB Identity()
	{
		AABB b;
		b.min[0] = FLT_MAX; 
		b.min[1] = FLT_MAX; 
		b.min[2] = FLT_MAX; 
		b.max[0] = -FLT_MAX+1;
		b.max[1] = -FLT_MAX+1;
		b.max[2] = -FLT_MAX+1;
		return b;
	}

	CUDA_FUNC_IN bool Intersect_FMA(const float3& I, const float3& OI, float* min = 0, float* max = 0) const
	{
		float tx1 = minV.x * I.x - OI.x;
		float tx2 = maxV.x * I.x - OI.x;
		float ty1 = minV.y * I.y - OI.y;
		float ty2 = maxV.y * I.y - OI.y;
		float tz1 = minV.z * I.z - OI.z;
		float tz2 = maxV.z * I.z - OI.z;
		float mi = spanBeginKepler(tx1, tx2, ty1, ty2, tz1, tz2, 0);
		float ma = spanEndKepler  (tx1, tx2, ty1, ty2, tz1, tz2, FLT_MAX);
		bool b = ma > mi && ma > 0;
		if(min && b)
			*min = mi;
		if(max && b)
			*max = ma;
		return b;
	}

	CUDA_FUNC_IN bool Intersect(const float3& m_Dir, const float3& m_Ori, float* min = 0, float* max = 0) const
	{
		float tx1 = (minV.x - m_Ori.x) / m_Dir.x;
		float tx2 = (maxV.x - m_Ori.x) / m_Dir.x;
		float ty1 = (minV.y - m_Ori.y) / m_Dir.y;
		float ty2 = (maxV.y - m_Ori.y) / m_Dir.y;
		float tz1 = (minV.z - m_Ori.z) / m_Dir.z;
		float tz2 = (maxV.z - m_Ori.z) / m_Dir.z;
		float mi = spanBeginKepler(tx1, tx2, ty1, ty2, tz1, tz2, 0);
		float ma = spanEndKepler  (tx1, tx2, ty1, ty2, tz1, tz2, FLT_MAX);
		bool b = ma > mi && ma > 0;
		if(min && b)
			*min = mi;
		if(max && b)
			*max = ma;
		return b;
	}

	CUDA_FUNC_IN bool Intersect(const Ray& r, float* min = 0, float* max = 0) const
	{
		return Intersect(r.direction, r.origin, min, max);
	}
};