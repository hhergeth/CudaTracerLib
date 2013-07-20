#pragma once

#include "..\Math\vector.h"
#include "..\Math\AABB.h"
#include "e_Buffer.h"

CUDA_FUNC_IN float PhaseHG(const float3 &w, const float3 &wp, float g)
{
    float costheta = dot(w, wp);
    return 1.f / (4.f * PI) * (1.f - g*g) / powf(1.f + g*g - 2.f * g * costheta, 1.5f);
}

struct e_BaseVolumeRegion
{
public:
	AABB Box;
	CUDA_FUNC_IN bool inside(const float3& p) const
	{
		return Box.Contains(p);
	}
};

/*
struct qq_REG : public e_BaseVolumeRegion
{
public:
    CUDA_FUNC_IN bool IntersectP(const Ray &ray, float *t0, float *t1)
	{

	}

    CUDA_FUNC_IN float3 sigma_a(const float3& p, const float3& w, float time) const
	{

	}

    CUDA_FUNC_IN float3 sigma_s(const float3& p, const float3& w, float time) const
	{

	}

    CUDA_FUNC_IN float3 Lve(const float3& p, const float3& w, float time) const
	{

	}

    CUDA_FUNC_IN float p(const float3& p, const float3& w, const float3& wp, float time) const
	{

	}

    CUDA_FUNC_IN float3 sigma_t(const float3 &p, const float3 &wo, float time) const
	{

	}

    CUDA_FUNC_IN float3 tau(const Ray &ray, float step = 1.f, float offset = 0.5) const
	{

	}
};
*/

#define e_HomogeneousVolumeDensity_TYPE 1
struct e_HomogeneousVolumeDensity : public e_BaseVolumeRegion
{
public:
	e_HomogeneousVolumeDensity(const float sa, const float ss, float gg, const float emit, const AABB box)
	{
		e_BaseVolumeRegion::Box = box;
        WorldToVolume = float4x4::NewIdentity();
        sig_a = make_float3(sa);
        sig_s = make_float3(ss);
        g = gg;
        le = make_float3(emit);
	}

	e_HomogeneousVolumeDensity(const float3 sa, const float3 ss, float gg, const float emit, const AABB box, const float4x4 v2w)
	{
		e_BaseVolumeRegion::Box = box;
        WorldToVolume = v2w.Inverse();
        sig_a = sa;
        sig_s = ss;
        g = gg;
        le = make_float3(emit);
	}

    CUDA_FUNC_IN bool IntersectP(const Ray &ray, const float minT, const float maxT, float *t0, float *t1) const
	{
		Ray r = ray * WorldToVolume;
		bool b = e_BaseVolumeRegion::Box.Intersect(r, t0, t1);
		if(b)
		{
			*t0 = clamp(*t0, minT, maxT);
			*t1 = clamp(*t1, minT, maxT);
		}
		return b && *t1 > *t0 && *t1 > 0;
	}

    CUDA_FUNC_IN float3 sigma_a(const float3& p, const float3& w) const
	{
		return inside(p) ? sig_a : make_float3(0.0f);
	}

    CUDA_FUNC_IN float3 sigma_s(const float3& p, const float3& w) const
	{
		return inside(p) ? sig_s : make_float3(0.0f);
	}

    CUDA_FUNC_IN float3 Lve(const float3& p, const float3& w) const
	{
		return inside(p) ? le : make_float3(0.0f);
	}

    CUDA_FUNC_IN float p(const float3& p, const float3& wi, const float3& wo) const
	{/*
		*wo = make_float3(0);
		if(inside(p))
		{
			float sqrTerm = (1 - g * g) / (1 - g + 2 * g * rng.randomFloat());
			float cosTheta = (1 + g * g - sqrTerm * sqrTerm) / (2 * g);
			float sinTheta = sqrtf(1.0f-cosTheta*cosTheta);
			float r = rng.randomFloat();
			float sinPhi = sinf(2*PI*r), cosPhi = cosf(2*PI*r);
			float3 r2 = make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
			*wo = Onb(-1.0f * wi).localToworld(r2);
			return 1;
		}
		else return 0;*/
		return inside(p) ? PhaseHG(wi, wo, g) : 0;
	}

    CUDA_FUNC_IN float3 sigma_t(const float3 &p, const float3 &wo) const
	{
		return inside(p) ? (sig_s + sig_a) : make_float3(0.0f);
	}

    CUDA_FUNC_IN float3 tau(const Ray &ray, const float minT, const float maxT, float step = 1.f, float offset = 0.5) const
	{
		float t0, t1;
		if(!IntersectP(ray, minT, maxT, &t0, &t1))
			return make_float3(0.0f);
		return make_float3(length(ray(t0) - ray(t1))) * (sig_a + sig_s);
	}

	TYPE_FUNC(e_HomogeneousVolumeDensity)
public:
	float3 sig_a, sig_s, le;
	float g;
	float4x4 WorldToVolume;
};

template<typename Density> struct e_DensityContainer : public e_BaseVolumeRegion
{
	e_DensityContainer(const float3& sa, const float3& ss, float gg, const float3& emit, const AABB& box, const float4x4& v2w, const Density& d)
		: m_sDensity(d)
	{
		e_BaseVolumeRegion::Box = box;
        WorldToVolume = float4x4::NewIdentity();
        sig_a = sa;
        sig_s = ss;
        g = gg;
        le = emit;
	}

    CUDA_FUNC_IN bool IntersectP(const Ray &ray, const float minT, const float maxT, float *t0, float *t1) const
	{
		Ray r = ray * WorldToVolume;
		bool b = e_BaseVolumeRegion::Box.Intersect(r, t0, t1);
		if(b)
		{
			*t0 = clamp(*t0, minT, maxT);
			*t1 = clamp(*t1, minT, maxT);
		}
		return b && *t1 > *t0 && *t1 > 0;
	}

    CUDA_FUNC_IN float3 sigma_a(const float3& p, const float3& w) const
	{
		return m_sDensity.f(WorldToVolume * p) * sig_a;
	}

    CUDA_FUNC_IN float3 sigma_s(const float3& p, const float3& w) const
	{
		return m_sDensity.f(WorldToVolume * p) * sig_s;
	}

    CUDA_FUNC_IN float3 Lve(const float3& p, const float3& w) const
	{
		return m_sDensity.f(WorldToVolume * p) * le;
	}

    CUDA_FUNC_IN float p(const float3& p, const float3& w, const float3& wp) const
	{
		return PhaseHG(w, wp, g);
	}

    CUDA_FUNC_IN float3 sigma_t(const float3 &p, const float3 &wo) const
	{
		return m_sDensity.f(WorldToVolume * p) * (sig_a + sig_s);
	}
protected:
	Density m_sDensity;
	float3 sig_a, sig_s, le;
    float g;
	float4x4 WorldToVolume;
};

template<typename Density> struct e_DensityContainerAnalytic : public e_DensityContainer<Density>
{
	e_DensityContainerAnalytic(const float3& sa, const float3& ss, float gg, const float3& emit, const AABB& box, const float4x4& v2w, const Density& d)
		: e_DensityContainer<Density>(sa, ss, gg, emit, box, v2w, d)
	{

	}

	CUDA_FUNC_IN float3 tau(const Ray &ray, float minT, float maxT, float stepSize = 1.f, float offset = 0.5) const
	{
		float3 start = WorldToVolume * ray(minT), end = WorldToVolume * ray(maxT);
		return m_sDensity.tau(start, end);
	}
};

template<typename Density> struct e_DensityContainerNumeric : public e_DensityContainer<Density>
{
	e_DensityContainerNumeric(const float3& sa, const float3& ss, float gg, const float3& emit, const AABB& box, const float4x4& v2w, const Density& d)
		: e_DensityContainer<Density>(sa, ss, gg, emit, box, v2w, d)
	{

	}

	CUDA_FUNC_IN float3 tau(const Ray &ray, float minT, float maxT, float stepSize = 1.f, float offset = 0.5) const
	{
		float t0, t1;
		float length = ::length(ray.direction);
		if (length == 0.f)
			return make_float3(0);
		Ray rn(ray.origin, ray.direction / length);
		if (!IntersectP(rn, minT * length, maxT * length, &t0, &t1))
			return make_float3(0);
		float3 tau = make_float3(0);
		t0 += offset * stepSize;
		while (t0 < t1)
		{
			tau += sigma_t(rn(t0), -rn.direction);
			t0 += stepSize;
		}
		return tau * stepSize;
	}
};

struct e_SphereDensity
{
	float4x4 mat;
	e_SphereDensity(float3 c, float r)
	{
		mat = float4x4::Translate(-c) * float4x4::Scale(1.0f / r);
	}

	CUDA_FUNC_IN float f(const float3& pObj) const
	{
		return 1.0f - clamp(length(mat * pObj), 0.0f, 1.0f);
	}/*
	CUDA_FUNC_IN float3 tau(const float3& start, const float3& end) const
	{

	}*/
};

#define e_SphereVolumeDensity_TYPE 2
struct e_SphereVolumeDensity : public e_DensityContainerNumeric<e_SphereDensity>
{
	e_SphereVolumeDensity(float3 center, float r, const float3& sa, const float3& ss, float gg, const float3& emit)
		: e_DensityContainerNumeric<e_SphereDensity>(sa, ss, gg, emit, AABB(center - make_float3(r), center + make_float3(r)), float4x4::Identity(), e_SphereDensity(center, r))
	{

	}

	e_SphereVolumeDensity(float3 center, float r, float sa, float ss, float gg, float emit)
		: e_DensityContainerNumeric<e_SphereDensity>(make_float3(sa), make_float3(ss), gg, make_float3(emit), AABB(center - make_float3(r), center + make_float3(r)), float4x4::Identity(), e_SphereDensity(center, r))
	{

	}

	TYPE_FUNC(e_SphereVolumeDensity)
};

struct e_VolumeRegion
{
#define CALL_TYPE(t,f,r) \
	case t##_TYPE : \
		r ((t*)Data)->f; \
		break;
#define CALL_FUNC(r,f) \
	switch (type) \
	{ \
		CALL_TYPE(e_HomogeneousVolumeDensity, f, r) \
		CALL_TYPE(e_SphereVolumeDensity, f, r) \
	}
private:
	unsigned char Data[256];
	unsigned int type;
public:
	CUDA_FUNC_IN e_VolumeRegion()
	{
		type = 0;
	}

	template<typename T> void Set(T& val)
	{
		*(T*)Data = val;
		type = T::TYPE();
	}

	CUDA_FUNC_IN AABB WorldBound()
	{
		return ((e_BaseVolumeRegion*)Data)->Box;
	}

	CUDA_FUNC_IN bool IntersectP(const Ray &ray, const float minT, const float maxT, float *t0, float *t1) const
	{
		CALL_FUNC(return, IntersectP(ray, minT, maxT, t0, t1))
	}

    CUDA_FUNC_IN float3 sigma_a(const float3& p, const float3& w) const
	{
		CALL_FUNC(return, sigma_a(p, w))
	}

    CUDA_FUNC_IN float3 sigma_s(const float3& p, const float3& w) const
	{
		CALL_FUNC(return, sigma_s(p, w))
	}

    CUDA_FUNC_IN float3 Lve(const float3& p, const float3& w) const
	{
		CALL_FUNC(return, Lve(p, w))
	}

    CUDA_FUNC_IN float p(const float3& _p, const float3& wi, const float3& wo) const
	{
		CALL_FUNC(return, p(_p, wi, wo))
	}

    CUDA_FUNC_IN float3 sigma_t(const float3 &p, const float3 &wo) const
	{
		CALL_FUNC(return, sigma_t(p, wo))
	}

    CUDA_FUNC_IN float3 tau(const Ray &ray, float minT, float maxT, float step = 1.f, float offset = 0.5) const
	{
		CALL_FUNC(return, tau(ray, minT, maxT, step, offset))
	}
#undef CALL_FUNC
#undef CALL_TYPE
};

struct e_KernelAggregateVolume
{
public:
	unsigned int m_uVolumeCount;
	e_VolumeRegion* m_pVolumes;
	AABB box;
public:
	CUDA_FUNC_IN e_KernelAggregateVolume()
	{

	}
	e_KernelAggregateVolume(e_Stream<e_VolumeRegion>* D)
	{
		m_uVolumeCount = D->UsedElements().getLength();
		m_pVolumes = D->getKernelData().Data;
		box = AABB::Identity();
		for(int i = 0; i < m_uVolumeCount; i++)
			box.Enlarge(D->operator()(i)->WorldBound());
	}

	//Calculates the intersection of the ray with the bound of the volume
    CUDA_FUNC_IN bool IntersectP(const Ray &ray, const float minT, const float maxT, float *t0, float *t1) const
	{
		*t0 = FLT_MAX;
		*t1 = -FLT_MAX;
		for(int i = 0; i < m_uVolumeCount; i++)
		{
			float a, b;
			if(m_pVolumes[i].IntersectP(ray, minT, maxT, &a, &b))
			{
				*t0 = MIN(*t0, a);
				*t1 = MAX(*t1, b);
			}
		}
		return (*t0 < *t1);
	}

	//The probability that light is abosrbed per unit distance
    CUDA_FUNC_IN float3 sigma_a(const float3& p, const float3& w) const
	{
		float3 s = make_float3(0);
		for(int i = 0; i < m_uVolumeCount; i++)
			s += m_pVolumes[i].sigma_a(p, w);
		return s;
	}

	//The probability that light is scattered per unit distance
    CUDA_FUNC_IN float3 sigma_s(const float3& p, const float3& w) const
	{
		float3 s = make_float3(0);
		for(int i = 0; i < m_uVolumeCount; i++)
			s += m_pVolumes[i].sigma_s(p, w);
		return s;
	}

    CUDA_FUNC_IN float3 Lve(const float3& p, const float3& w) const
	{
		float3 s = make_float3(0);
		for(int i = 0; i < m_uVolumeCount; i++)
			s += m_pVolumes[i].Lve(p, w);
		return s;
	}

	//Calculates the phase function at a given point
    CUDA_FUNC_IN float p(const float3& p, const float3& w, const float3& wo) const
	{
		float ph = 0, sumWt = 0;
		for(int i = 0; i < m_uVolumeCount; i++)
		{
			float wt = y(m_pVolumes[i].sigma_s(p, w));
			if(wt != 0)
			{
				sumWt += wt;
				ph += wt * m_pVolumes[i].p(p, w, wo);
			}
		}
		return sumWt != 0 ? ph / sumWt : 0.0f;
	}

	//Combined sigmas
    CUDA_FUNC_IN float3 sigma_t(const float3 &p, const float3 &wo) const
	{
		float3 s = make_float3(0);
		for(int i = 0; i < m_uVolumeCount; i++)
			s += m_pVolumes[i].sigma_t(p, wo);
		return s;
	}

	//Calculates the volumes optical thickness along a ray in the volumes bounds
    CUDA_FUNC_IN float3 tau(const Ray &ray, const float minT, const float maxT, float step = 1.f, float offset = 0.5) const
	{
		float3 s = make_float3(0);
		for(int i = 0; i < m_uVolumeCount; i++)
			s += m_pVolumes[i].tau(ray, minT, maxT, step, offset);
		return s;
	}

	CUDA_FUNC_IN bool HasVolumes()
	{
		return m_uVolumeCount > 0;
	}
};