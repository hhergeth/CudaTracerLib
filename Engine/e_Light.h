#pragma once

#include "..\Base\CudaRandom.h"
#include "..\Math\vector.h"
#include "..\Math\AABB.h"
#include "e_ShapeSet.h"
#include "e_KernelDynamicScene.h"

#define MAX_SHAPE_LENGTH 32

struct e_VisibilitySegment
{
	Ray r;
	float tmin;
	float tmax;

	CUDA_FUNC_IN e_VisibilitySegment()
	{

	}

	CUDA_FUNC_IN void SetSegment(const float3& o, float offo, const float3& t, float offt)
	{
		r.direction = normalize(t - o);
		r.origin = o;
		tmin = offo;
		tmax = length(t - o) + offt;//trust in the compiler :D
	}

	CUDA_FUNC_IN void SetRay(const float3& o, float offo, const float3& d)
	{
		r.direction = d;
		r.origin = o;
		tmin = offo;
		tmax = FLT_MAX;
	}

	CUDA_FUNC_IN bool IsValidHit(float thit) const
	{
		return tmin <= thit && thit <= tmax;
	}
};

struct e_LightBase
{
	bool IsDelta;

	e_LightBase(bool d)
		: IsDelta(d)
	{
	}
};

#define e_PointLight_TYPE 1
struct e_PointLight : public e_LightBase
{
	float3 lightPos;
    float3 Intensity;

	e_PointLight(float3 p, float3 L)
		: e_LightBase(true), lightPos(p), Intensity(L)
	{

	}

	CUDA_FUNC_IN float3 Power(const e_KernelDynamicScene& scene) const
	{
		return 4.f * PI * Intensity;
	}

	CUDA_FUNC_IN float Pdf(const e_KernelDynamicScene& scene, const float3 &p, const float3 &wi) const
	{
		return 0.0f;
	}

	CUDA_FUNC_IN float3 Sample_L(const e_KernelDynamicScene& scene, const LightSample &ls, float u1, float u2, Ray *ray, float3 *Ns, float *pdf) const
	{
		*ray = Ray(lightPos, UniformSampleSphere(ls.uPos[0], ls.uPos[1]));
		*Ns = ray->direction;
		*pdf = UniformSpherePdf();
		return Intensity;
	}

	CUDA_FUNC_IN float3 Sample_L(const e_KernelDynamicScene& scene, const float3& p, const LightSample& ls, float* pdf, e_VisibilitySegment* seg) const
	{
		seg->SetSegment(p, 0, lightPos, 0);
		*pdf = 1.0f;
		return Intensity / DistanceSquared(lightPos, p);
	}

	CUDA_FUNC_IN float3 Le(const e_KernelDynamicScene& scene, const Ray& r) const
	{
		return make_float3(0.0f);
	}
	
	CUDA_FUNC_IN float3 L(const float3 &p, const float3 &n, const float3 &w) const
	{
		return make_float3(0);
	}

	AABB getBox(float eps) const
	{
		return AABB(lightPos - make_float3(eps), lightPos + make_float3(eps));
	}
public:
	static const unsigned int TYPE;
};

#define e_DiffuseLight_TYPE 2
struct e_DiffuseLight : public e_LightBase
{
	float3 Lemit;
    ShapeSet<MAX_SHAPE_LENGTH> shapeSet;

	e_DiffuseLight(float3 L, ShapeSet<MAX_SHAPE_LENGTH>& s)
		: e_LightBase(false), shapeSet(s), Lemit(L)
	{

	}

	CUDA_FUNC_IN float3 Power(const e_KernelDynamicScene& scene) const
	{
		return Lemit * shapeSet.Area() * PI;
	}

	CUDA_FUNC_IN float Pdf(const e_KernelDynamicScene& scene, const float3 &p, const float3 &wi) const
	{
		return shapeSet.Pdf(p, wi, scene.m_sBVHIntData.Data);
	}

	CUDA_FUNC_IN float3 Sample_L(const e_KernelDynamicScene& scene, const LightSample &ls, float u1, float u2, Ray *ray, float3 *Ns, float *pdf) const
	{
		float3 org = shapeSet.Sample(ls, Ns, scene.m_sBVHIntData.Data);
		float3 dir = UniformSampleSphere(u1, u2);
		if (dot(dir, *Ns) < 0.)
			dir *= -1.f;
		*ray = Ray(org, dir);
		*pdf = shapeSet.Pdf(org) * INV_TWOPI;
		float3 Ls = L(org, *Ns, dir);
		return Ls;
	}

	CUDA_FUNC_IN float3 Sample_L(const e_KernelDynamicScene& scene, const float3& p, const LightSample& ls, float* pdf, e_VisibilitySegment* seg) const
	{
		float3 ns;
		float3 ps = shapeSet.Sample(p, ls, &ns, scene.m_sBVHIntData.Data);
		seg->SetSegment(p, 0, ps, 0);
		*pdf = shapeSet.Pdf(p, seg->r.direction, scene.m_sBVHIntData.Data);
		return L(ps, ns, -seg->r.direction);
	}

	CUDA_FUNC_IN float3 Le(const e_KernelDynamicScene& scene, const Ray& r) const
	{
		return make_float3(0.0f);
	}

	CUDA_FUNC_IN float3 L(const float3 &p, const float3 &n, const float3 &w) const
	{
        return dot(n, w) > 0.f ? Lemit : make_float3(0);
    }

	AABB getBox(float eps) const
	{
		return shapeSet.getBox();
	}
public:
	static const unsigned int TYPE;
};

#define e_DistantLight_TYPE 3
struct e_DistantLight : public e_LightBase
{
	float3 lightDir;
    float3 _L;
	Onb sys;

	e_DistantLight(float3 l, float3 d)
		: e_LightBase(true), lightDir(d), _L(l), sys(d)
	{

	}

	CUDA_FUNC_IN float3 Power(const e_KernelDynamicScene& scene) const
	{
		float3 worldCenter = scene.m_sBox.Center();
		float worldRadius = Distance(scene.m_sBox.maxV, scene.m_sBox.minV) / 2.0f;
		return _L * PI * worldRadius * worldRadius;
	}

	CUDA_FUNC_IN float Pdf(const e_KernelDynamicScene& scene, const float3 &p, const float3 &wi) const
	{
		return 0.0f;
	}

	CUDA_FUNC_IN float3 Sample_L(const e_KernelDynamicScene& scene, const LightSample &ls, float u1, float u2, Ray *ray, float3 *Ns, float *pdf) const
	{
		float3 worldCenter = scene.m_sBox.Center();
		float worldRadius = Distance(scene.m_sBox.maxV, scene.m_sBox.minV) / 2.0f;
		float d1, d2;
		ConcentricSampleDisk(ls.uPos[0], ls.uPos[1], &d1, &d2);
		float3 Pdisk = worldCenter + worldRadius * (d1 * sys.m_binormal + d2 * sys.m_tangent);
		*ray = Ray(Pdisk + worldRadius * lightDir, -1.0f * lightDir);
		*Ns = -1.0f * lightDir;
		*pdf = 1.f / (PI * worldRadius * worldRadius);
		return _L;
	}

	CUDA_FUNC_IN float3 Sample_L(const e_KernelDynamicScene& scene, const float3& p, const LightSample& ls, float* pdf, e_VisibilitySegment* seg) const
	{
		seg->SetRay(p, 0, lightDir);
		*pdf = 1.0f;
		return _L;
	}
	
	CUDA_FUNC_IN float3 L(const float3 &p, const float3 &n, const float3 &w) const
	{
		return make_float3(0);
	}

	CUDA_FUNC_IN float3 Le(const e_KernelDynamicScene& scene, const Ray& r) const
	{
		return make_float3(0.0f);
	}
	
	AABB getBox(float eps) const
	{
		return AABB(make_float3(0), make_float3(0));
	}
public:
	static const unsigned int TYPE;
};

#define e_SpotLight_TYPE 4
struct e_SpotLight : public e_LightBase
{
    float3 lightPos;
    float3 Intensity;
    float cosTotalWidth, cosFalloffStart;
	Onb sys;

	e_SpotLight(float3 p, float3 t, float3 L, float width, float fall)
		: e_LightBase(true), lightPos(p), Intensity(L)
	{
		cosTotalWidth = cosf(Radians(width));
		cosFalloffStart = cosf(Radians(fall));
		sys = Onb(t - p);
	}

	CUDA_FUNC_IN float3 Power(const e_KernelDynamicScene& scene) const
	{
		return Intensity * 2.f * PI * (1.f - .5f * (cosFalloffStart + cosTotalWidth));
	}

	CUDA_FUNC_IN float Pdf(const e_KernelDynamicScene& scene, const float3 &p, const float3 &wi) const
	{
		return 0.0f;
	}

	CUDA_FUNC_IN float3 Sample_L(const e_KernelDynamicScene& scene, const LightSample &ls, float u1, float u2, Ray *ray, float3 *Ns, float *pdf) const
	{
		float3 v = UniformSampleCone(ls.uPos[0], ls.uPos[1], cosTotalWidth);
		*ray = Ray(lightPos, sys.localToworld(v));
		*Ns = ray->direction;
		*pdf = UniformConePdf(cosTotalWidth);
		return Intensity * Falloff(v);
	}

	CUDA_FUNC_IN float3 Sample_L(const e_KernelDynamicScene& scene, const float3& p, const LightSample& ls, float* pdf, e_VisibilitySegment* seg) const
	{
		seg->SetSegment(p, 0, lightPos, 0);
		*pdf = 1.0f;
		return Intensity * Falloff(sys.worldTolocal(-seg->r.direction)) / DistanceSquared(lightPos, p);
	}
	
	CUDA_FUNC_IN float3 L(const float3 &p, const float3 &n, const float3 &w) const
	{
		return make_float3(0);
	}

	CUDA_FUNC_IN float3 Le(const e_KernelDynamicScene& scene, const Ray& r) const
	{
		return make_float3(0.0f);
	}
	
	AABB getBox(float eps) const
	{
		return AABB(lightPos - make_float3(eps), lightPos + make_float3(eps));
	}
private:
	CUDA_FUNC_IN float Falloff(const float3 &w) const
	{
		float3 wl = normalize(w);
		float costheta = wl.z;
		if (costheta < cosTotalWidth)     return 0.;
		if (costheta > cosFalloffStart)   return 1.;
		// Compute falloff inside spotlight cone
		float delta = (costheta - cosTotalWidth) / (cosFalloffStart - cosTotalWidth);
		return delta*delta*delta*delta;
	}
public:
	static const unsigned int TYPE;
};

struct e_KernelLight
{
private:
	unsigned char Data[sizeof(ShapeSet<MAX_SHAPE_LENGTH>) * 4];
	unsigned int type;
#define CALL_TYPE(t,f,r) \
	case t##_TYPE : \
		r ((t*)Data)->f; \
		break;
#define CALL_FUNC(r,f) \
	switch (type) \
	{ \
		CALL_TYPE(e_PointLight, f, r) \
		CALL_TYPE(e_DiffuseLight, f, r) \
		CALL_TYPE(e_DistantLight, f, r) \
		CALL_TYPE(e_SpotLight, f, r) \
	}
public:
	template<typename T> void Set(T& val)
	{
		*(T*)Data = val;
		type = T::TYPE;
	}

	CUDA_FUNC_IN float3 Power(const e_KernelDynamicScene& scene) const
	{
		CALL_FUNC(return, Power(scene))
	}

	CUDA_FUNC_IN float Pdf(const e_KernelDynamicScene& scene, const float3 &p, const float3 &wi) const
	{
		CALL_FUNC(return, Pdf(scene, p, wi))
	}

	CUDA_FUNC_IN float3 Sample_L(const e_KernelDynamicScene& scene, const LightSample &ls, float u1, float u2, Ray *ray, float3 *Ns, float *pdf) const
	{
		CALL_FUNC(return, Sample_L(scene, ls, u1, u2, ray, Ns, pdf))
	}

	CUDA_FUNC_IN float3 Sample_L(const e_KernelDynamicScene& scene, const float3& p, const LightSample& ls, float* pdf, e_VisibilitySegment* seg) const
	{
		CALL_FUNC(return, Sample_L(scene, p, ls, pdf, seg))
	}

	CUDA_FUNC_IN float3 Le(const e_KernelDynamicScene& scene, const Ray& r) const
	{
		CALL_FUNC(return, Le(scene, r))
	}
	
	CUDA_FUNC_IN float3 L(const float3 &p, const float3 &n, const float3 &w) const
	{
		CALL_FUNC(return, L(p, n, w))
	}

	CUDA_FUNC_IN bool IsDeltaLight() const
	{
		return ((e_LightBase*)Data)->IsDelta;
	}

	AABB getBox(float eps) const
	{
		CALL_FUNC(return, getBox(eps))
	}

	template<typename T> T* As()
	{
		return (T*)Data;
	}

#undef CALL_FUNC
#undef CALL_TYPE
};

