#pragma once

#include "..\Base\CudaRandom.h"
#include "..\Math\vector.h"
#include "..\Math\AABB.h"

enum e_LightType  : unsigned int { LT_Directional, LT_Sphere, LT_Directed};

struct e_KernelLight
{
	float3 m_cPower;
	AABB box;
	//union
	//{
		//struct//LT_Directional
		//{
			float3 m_vDirection;
			float3 m_vOrigin;
			float3 m_vSpan;
		//};
		//struct//LT_Sphere
		//{
			float3 m_vSphereOrigin;
			float m_fRadius;
			float m_fRadSqr;
		//};
		//struct//LT_Directed
		//{
			float3 m_vOriginS;
			float3 m_vSpanS;
			float3 m_vOriginD;
			float3 m_vSpanD;
			float3 m_vNormal;
		//}
	//};
	e_LightType Type;
	CUDA_FUNC_IN float3 L(const float3 &p, const float3 &n, const float3 &w) const 
	{
		return dot(n, w) > 0.f ? m_cPower : make_float3(0);
	}
	CUDA_ONLY_FUNC float3 Sample_L(CudaRNG& rng, Ray* photonRay, float3* nor = 0, float* pdf = 0) const
	{
		if(pdf)
			*pdf = 1;
		float3 n;
		Ray r;
		if(Type == LT_Directional)
		{
			float3 o = m_vOrigin + m_vSpan * make_float3(rng.randomFloat(), rng.randomFloat(), rng.randomFloat());
			r = Ray(o, SampleCosineHemisphere(m_vDirection, rng.randomFloat(), rng.randomFloat()));
			n = m_vDirection;
		}
		else if(Type == LT_Sphere)
		{
			float3 d = normalize(make_float3(rng.randomFloat(), rng.randomFloat(), rng.randomFloat()) * 2.0f - 1.0f);
			r =  Ray(m_vSphereOrigin + d * m_fRadius * 1.01f, d);
			n = d;
		}
		else if(Type == LT_Directed)
		{
			float3 o = m_vOriginS + m_vSpanS * make_float3(rng.randomFloat(), rng.randomFloat(), rng.randomFloat());
			float3 h = m_vOriginD + m_vSpanD * make_float3(rng.randomFloat(), rng.randomFloat(), rng.randomFloat());
			float3 d = normalize(h - o);
			r =  Ray(o, d);
			n = m_vNormal;
		}
		if(nor)
			*nor = n;
		*photonRay = r;
		return this->L(r.origin, n, r.direction);
	}
	CUDA_ONLY_FUNC unsigned int CheckHit(float3& p)
	{
		const float eps = 0.05f, e = 1.0f + eps;
		if(Type == LT_Directional)
		{
			float3 c = (p - m_vOrigin) / m_vSpan;
			return c.x < e && c.x > -eps && c.y < e && c.y > -eps && c.z < e && c.z > -eps;
		}
		else if(Type == LT_Sphere)
		{
			float3 c = p - m_vSphereOrigin;
			return dot(c, c) < m_fRadSqr * e;
		}
	}
	CUDA_ONLY_FUNC float3 SampleRandomPoint(CudaRNG& rng)
	{
		if(Type == LT_Directional)
			return m_vOrigin + m_vSpan * make_float3(rng.randomFloat(), rng.randomFloat(), rng.randomFloat());
			//return m_vOrigin + m_vSpan * make_float3(0.5f,0,0.5f);
		else if(Type == LT_Sphere)
		{
			float3 d = normalize(make_float3(rng.randomFloat(), rng.randomFloat(), rng.randomFloat()) * 2.0f - 1.0f);
			return m_vSphereOrigin + d * m_fRadius * 1.01f;
		}
	}
};

class e_Light
{
protected:
	float3 m_cPower;
public:
	e_Light(float3 a_Power)
	{
		m_cPower = a_Power;
	}
	virtual e_KernelLight getKernelData() = 0;
	virtual AABB getBox() = 0;
};

class e_DirectionalLight : public e_Light
{
private:
	float3 m_vDirection;
	float3 m_vOrigin;
	float3 m_vSpan;
public:
	e_DirectionalLight(AABB& box, float3& dir, float3& col)
		: e_Light(col)
	{
		box = box.Inflate();
		m_vDirection = dir;
		m_vOrigin = box.minV;
		m_vSpan = box.maxV - box.minV;
	}
	virtual e_KernelLight getKernelData()
	{
		e_KernelLight r;
		r.Type = LT_Directional;
		r.m_cPower = m_cPower;
		r.m_vDirection = m_vDirection;
		r.m_vOrigin = m_vOrigin;
		r.m_vSpan = m_vSpan;
		r.box = getBox();
		return r;
	}
	virtual AABB getBox()
	{
		return AABB(m_vOrigin - make_float3(0.1f), m_vOrigin + m_vSpan + make_float3(0.1f));
	}
};

class e_SphereLight : public e_Light
{
private:
	float3 m_vPoint;
	float m_fRadius;
public:
	e_SphereLight(float3 pos, float rad, float3& col)
		: e_Light(col)
	{
		m_vPoint = pos;
		m_fRadius = rad;
	}
	virtual e_KernelLight getKernelData()
	{
		e_KernelLight r;
		r.Type = LT_Sphere;
		r.m_cPower = m_cPower;
		r.m_vSphereOrigin = m_vPoint;
		r.m_fRadius = m_fRadius;
		r.m_fRadSqr = m_fRadius * m_fRadius;
		r.box = getBox();
		return r;
	}
	virtual AABB getBox()
	{
		float3 r = make_float3(m_fRadius);
		return AABB(m_vPoint - r, m_vPoint + r);
	}
};

class e_DirectedLight : public e_Light
{
private:
	unsigned int type;
	e_Node* m_pTarget0;
	AABB m_pTarget1;
	float3 m_vOrigin;
	float3 m_vSpan;
public:
	e_DirectedLight(e_Node* N, float3& a_Origin, float3& a_Span, float3& col)
		: e_Light(col)
	{
		type = 1;
		m_pTarget0 = N;
		m_vOrigin = a_Origin;
		m_vSpan = a_Span;
	}
	e_DirectedLight(AABB& dest, float3& a_Origin, float3& a_Span, float3& col)
		: e_Light(col)
	{
		type = 2;
		m_pTarget1 = dest;
		m_vOrigin = a_Origin;
		m_vSpan = a_Span;
	}
	virtual e_KernelLight getKernelData()
	{
		e_KernelLight r;
		r.m_cPower = m_cPower;
		r.Type = LT_Directed;
		r.m_vOriginS = m_vOrigin;
		r.m_vSpanS = m_vSpan;
		AABB b = type == 1 ? m_pTarget0->getWorldBox() : m_pTarget1;
		r.m_vOriginD = b.minV; 
		r.m_vSpanD = b.maxV - b.minV;
		r.box = getBox();
		float3 d2 = b.Center() - r.box.Center(), d= fabsf(d2);
		if(d.x > d.y && d.x > d.z)
			r.m_vNormal = make_float3(signf(d2.x),0,0);
		else if(d.y > d.z)
			r.m_vNormal = make_float3(0,signf(d2.y),0);
		else r.m_vNormal = make_float3(0,0,signf(d2.z));
		return r;
	}
	virtual AABB getBox()
	{
		return AABB(m_vOrigin, m_vOrigin + m_vSpan);
	}
};

inline int maxLightSize()
{
	return MAX(sizeof(e_DirectionalLight), sizeof(e_SphereLight), sizeof(e_DirectedLight));
}