#pragma once

#include "..\Base\CudaRandom.h"
#include <MathTypes.h>
#include "e_ShapeSet.h"
#include "e_KernelDynamicScene.h"

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
    Spectrum Intensity;
	float radius;

	e_PointLight(float3 p, Spectrum L, float r = 0)
		: e_LightBase(true), lightPos(p), Intensity(L), radius(r)
	{

	}

	CUDA_FUNC_IN Spectrum Power(const e_KernelDynamicScene& scene) const
	{
		return 4.f * PI * Intensity;
	}

	CUDA_FUNC_IN float Pdf(const e_KernelDynamicScene& scene, const float3 &p, const float3 &wi) const
	{
		return 0.0f;
	}

	CUDA_DEVICE CUDA_HOST Spectrum Sample_L(const e_KernelDynamicScene& scene, const LightSample &ls, float u1, float u2, Ray *ray, float3 *Ns, float *pdf) const;

	CUDA_FUNC_IN Spectrum Sample_L(const e_KernelDynamicScene& scene, const float3& p, const LightSample& ls, float* pdf, e_VisibilitySegment* seg) const
	{
		seg->SetSegment(p, 0, lightPos, 0);
		*pdf = 1.0f;
		return Intensity / DistanceSquared(lightPos, p);
	}

	CUDA_FUNC_IN Spectrum Le(const e_KernelDynamicScene& scene, const Ray& r) const
	{
		return make_float3(0.0f);
	}
	
	CUDA_FUNC_IN Spectrum L(const float3 &p, const float3 &n, const float3 &w) const
	{
		return make_float3(0);
	}

	AABB getBox(float eps) const
	{
		return AABB(lightPos - make_float3(MAX(eps, radius)), lightPos + make_float3(MAX(eps, radius)));
	}
	
	TYPE_FUNC(e_PointLight)
};

#define e_DiffuseLight_TYPE 2
struct e_DiffuseLight : public e_LightBase
{
	CUDA_ALIGN(16) Spectrum Lemit;
    CUDA_ALIGN(16) ShapeSet shapeSet;

	e_DiffuseLight(Spectrum L, ShapeSet& s)
		: e_LightBase(false), shapeSet(s), Lemit(L)
	{

	}

	CUDA_FUNC_IN Spectrum Power(const e_KernelDynamicScene& scene) const
	{
		return Lemit * shapeSet.Area() * PI;
	}

	CUDA_FUNC_IN float Pdf(const e_KernelDynamicScene& scene, const float3 &p, const float3 &wi) const
	{
		return shapeSet.Pdf(p, wi, scene.m_sBVHIntData.Data);
	}

	CUDA_DEVICE CUDA_HOST Spectrum Sample_L(const e_KernelDynamicScene& scene, const LightSample &ls, float u1, float u2, Ray *ray, float3 *Ns, float *pdf) const;

	CUDA_DEVICE CUDA_HOST Spectrum Sample_L(const e_KernelDynamicScene& scene, const float3& p, const LightSample& ls, float* pdf, e_VisibilitySegment* seg) const;

	CUDA_FUNC_IN Spectrum Le(const e_KernelDynamicScene& scene, const Ray& r) const
	{
		return Spectrum(0.0f);
	}

	CUDA_FUNC_IN Spectrum L(const float3 &p, const float3 &n, const float3 &w) const
	{
        return dot(n, w) > 0.f ? Lemit : Spectrum(0.0f);
    }

	AABB getBox(float eps) const
	{
		return shapeSet.getBox();
	}
	
	TYPE_FUNC(e_DiffuseLight)
};

#define e_DistantLight_TYPE 3
struct e_DistantLight : public e_LightBase
{
	float3 lightDir;
    Spectrum _L;
	Frame sys;

	e_DistantLight(Spectrum l, float3 d)
		: e_LightBase(true), lightDir(d), _L(l), sys(d)
	{

	}

	CUDA_DEVICE CUDA_HOST Spectrum Power(const e_KernelDynamicScene& scene) const;

	CUDA_FUNC_IN float Pdf(const e_KernelDynamicScene& scene, const float3 &p, const float3 &wi) const
	{
		return 0.0f;
	}

	CUDA_DEVICE CUDA_HOST Spectrum Sample_L(const e_KernelDynamicScene& scene, const LightSample &ls, float u1, float u2, Ray *ray, float3 *Ns, float *pdf) const;

	CUDA_FUNC_IN Spectrum Sample_L(const e_KernelDynamicScene& scene, const float3& p, const LightSample& ls, float* pdf, e_VisibilitySegment* seg) const
	{
		seg->SetRay(p, 0, lightDir);
		*pdf = 1.0f;
		return _L;
	}
	
	CUDA_FUNC_IN Spectrum L(const float3 &p, const float3 &n, const float3 &w) const
	{
		return Spectrum(0.0f);
	}

	CUDA_FUNC_IN Spectrum Le(const e_KernelDynamicScene& scene, const Ray& r) const
	{
		return Spectrum(0.0f);
	}
	
	AABB getBox(float eps) const
	{
		return AABB(make_float3(0), make_float3(0));
	}
	
	TYPE_FUNC(e_DistantLight)
};

#define e_SpotLight_TYPE 4
struct e_SpotLight : public e_LightBase
{
    float3 lightPos;
    Spectrum Intensity;
    float cosTotalWidth, cosFalloffStart;
	Frame sys;

	e_SpotLight(float3 p, float3 t, Spectrum L, float width, float fall)
		: e_LightBase(true), lightPos(p), Intensity(L)
	{
		cosTotalWidth = cosf(Radians(width));
		cosFalloffStart = cosf(Radians(fall));
		sys = Frame(t - p);
	}

	CUDA_FUNC_IN Spectrum Power(const e_KernelDynamicScene& scene) const
	{
		return Intensity * 2.f * PI * (1.f - .5f * (cosFalloffStart + cosTotalWidth));
	}

	CUDA_FUNC_IN float Pdf(const e_KernelDynamicScene& scene, const float3 &p, const float3 &wi) const
	{
		return 0.0f;
	}

	CUDA_DEVICE CUDA_HOST Spectrum Sample_L(const e_KernelDynamicScene& scene, const LightSample &ls, float u1, float u2, Ray *ray, float3 *Ns, float *pdf) const;

	CUDA_DEVICE CUDA_HOST Spectrum Sample_L(const e_KernelDynamicScene& scene, const float3& p, const LightSample& ls, float* pdf, e_VisibilitySegment* seg) const;
	
	CUDA_FUNC_IN Spectrum L(const float3 &p, const float3 &n, const float3 &w) const
	{
		return make_float3(0);
	}

	CUDA_FUNC_IN Spectrum Le(const e_KernelDynamicScene& scene, const Ray& r) const
	{
		return make_float3(0.0f);
	}
	
	AABB getBox(float eps) const
	{
		return AABB(lightPos - make_float3(eps), lightPos + make_float3(eps));
	}
	
	TYPE_FUNC(e_SpotLight)
private:
	CUDA_DEVICE CUDA_HOST float Falloff(const float3 &w) const;
};

#define e_InfiniteLight_TYPE 5
struct e_InfiniteLight : public e_LightBase
{
	//e_BufferReference<Distribution2D<4096, 4096>, Distribution2D<4096, 4096>> distribution;
	e_KernelMIPMap radianceMap;
	Distribution2D<4096, 4096>* pDist;

	CUDA_HOST e_InfiniteLight(const Spectrum& power, e_BufferReference<Distribution2D<4096, 4096>, Distribution2D<4096, 4096>>& d, e_BufferReference<e_MIPMap, e_KernelMIPMap>& mip);

	CUDA_DEVICE CUDA_HOST Spectrum Power(const e_KernelDynamicScene& scene) const;

	CUDA_DEVICE CUDA_HOST float Pdf(const e_KernelDynamicScene& scene, const float3 &p, const float3 &wi) const;

	CUDA_DEVICE CUDA_HOST Spectrum Sample_L(const e_KernelDynamicScene& scene, const LightSample &ls, float u1, float u2, Ray *ray, float3 *Ns, float *pdf) const;

	CUDA_DEVICE CUDA_HOST Spectrum Sample_L(const e_KernelDynamicScene& scene, const float3& p, const LightSample& ls, float* pdf, e_VisibilitySegment* seg) const;
	
	CUDA_FUNC_IN Spectrum L(const float3 &p, const float3 &n, const float3 &w) const
	{
		return Spectrum(0.0f);
	}

	CUDA_FUNC_IN Spectrum Le(const e_KernelDynamicScene& scene, const Ray& r) const
	{
		float s = SphericalPhi(r.direction) * INV_TWOPI, t = SphericalTheta(r.direction) * INV_PI;
		return radianceMap.Sample<Spectrum>(make_float2(s, t), 0);
	}
	
	AABB getBox(float eps) const
	{
		return AABB(-make_float3(1.0f / eps), make_float3(1.0f / eps));
	}

	TYPE_FUNC(e_InfiniteLight)
};

#define LGT_SIZE RND_16(DMAX5(sizeof(e_PointLight), sizeof(e_DiffuseLight), sizeof(e_DistantLight), sizeof(e_SpotLight), sizeof(e_InfiniteLight)))

CUDA_ALIGN(16) struct e_KernelLight
{
public:
	CUDA_ALIGN(16) unsigned char Data[LGT_SIZE];
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
		CALL_TYPE(e_InfiniteLight, f, r) \
	}
public:
	template<typename T> void SetData(const T& val)
	{
		memcpy(Data, &val, sizeof(T));
		type = T::TYPE();
	}

	CUDA_FUNC_IN Spectrum Power(const e_KernelDynamicScene& scene) const
	{
		CALL_FUNC(return, Power(scene))
	}

	CUDA_FUNC_IN float Pdf(const e_KernelDynamicScene& scene, const float3 &p, const float3 &wi) const
	{
		CALL_FUNC(return, Pdf(scene, p, wi))
	}

	CUDA_FUNC_IN Spectrum Sample_L(const e_KernelDynamicScene& scene, const LightSample &ls, float u1, float u2, Ray *ray, float3 *Ns, float *pdf) const
	{
		CALL_FUNC(return, Sample_L(scene, ls, u1, u2, ray, Ns, pdf))
	}

	CUDA_FUNC_IN Spectrum Sample_L(const e_KernelDynamicScene& scene, const float3& p, const LightSample& ls, float* pdf, e_VisibilitySegment* seg) const
	{
		CALL_FUNC(return, Sample_L(scene, p, ls, pdf, seg))
	}

	CUDA_FUNC_IN Spectrum Le(const e_KernelDynamicScene& scene, const Ray& r) const
	{
		CALL_FUNC(return, Le(scene, r))
	}
	
	CUDA_FUNC_IN Spectrum L(const float3 &p, const float3 &n, const float3 &w) const
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

