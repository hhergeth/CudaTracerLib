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
	e_KernelMIPMap radianceMap;
	Distribution2D<4096, 4096>* pDistDevice;
	Distribution2D<4096, 4096>* pDistHost;

	CUDA_HOST e_InfiniteLight(const Spectrum& power, e_StreamReference(char)& d, e_BufferReference<e_MIPMap, e_KernelMIPMap>& mip);

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
		float s = MonteCarlo::SphericalPhi(r.direction) * INV_TWOPI, t = MonteCarlo::SphericalTheta(r.direction) * INV_PI;
		return radianceMap.Sample(make_float2(s, t), 0);
	}
	
	AABB getBox(float eps) const
	{
		return AABB(-make_float3(1.0f / eps), make_float3(1.0f / eps));
	}

	TYPE_FUNC(e_InfiniteLight)
private:
	CUDA_FUNC_IN Distribution2D<4096, 4096>* dist() const
	{
#ifdef ISCUDA
		return pDistDevice;
#else
		return pDistHost;
#endif
	}
};

#define LGT_SIZE RND_16(DMAX5(sizeof(e_PointLight), sizeof(e_DiffuseLight), sizeof(e_DistantLight), sizeof(e_SpotLight), sizeof(e_InfiniteLight)))

CUDA_ALIGN(16) struct e_KernelLight
{
public:
	CUDA_ALIGN(16) unsigned char Data[LGT_SIZE];
	unsigned int type;
public:
	CUDA_FUNC_IN Spectrum Power(const e_KernelDynamicScene& scene) const
	{
		CALL_FUNC5(e_PointLight,e_DiffuseLight,e_DistantLight,e_SpotLight,e_InfiniteLight, Power(scene))
		return 0.0f;
	}

	CUDA_FUNC_IN float Pdf(const e_KernelDynamicScene& scene, const float3 &p, const float3 &wi) const
	{
		CALL_FUNC5(e_PointLight,e_DiffuseLight,e_DistantLight,e_SpotLight,e_InfiniteLight, Pdf(scene, p, wi))
		return 0.0f;
	}

	CUDA_FUNC_IN Spectrum Sample_L(const e_KernelDynamicScene& scene, const LightSample &ls, float u1, float u2, Ray *ray, float3 *Ns, float *pdf) const
	{
		CALL_FUNC5(e_PointLight,e_DiffuseLight,e_DistantLight,e_SpotLight,e_InfiniteLight, Sample_L(scene, ls, u1, u2, ray, Ns, pdf))
		return 0.0f;
	}

	CUDA_FUNC_IN Spectrum Sample_L(const e_KernelDynamicScene& scene, const float3& p, const LightSample& ls, float* pdf, e_VisibilitySegment* seg) const
	{
		CALL_FUNC5(e_PointLight,e_DiffuseLight,e_DistantLight,e_SpotLight,e_InfiniteLight, Sample_L(scene, p, ls, pdf, seg))
		return 0.0f;
	}

	CUDA_FUNC_IN Spectrum Le(const e_KernelDynamicScene& scene, const Ray& r) const
	{
		CALL_FUNC5(e_PointLight,e_DiffuseLight,e_DistantLight,e_SpotLight,e_InfiniteLight, Le(scene, r))
		return 0.0f;
	}
	
	CUDA_FUNC_IN Spectrum L(const float3 &p, const float3 &n, const float3 &w) const
	{
		CALL_FUNC5(e_PointLight,e_DiffuseLight,e_DistantLight,e_SpotLight,e_InfiniteLight, L(p, n, w))
		return 0.0f;
	}

	CUDA_FUNC_IN bool IsDeltaLight() const
	{
		return ((e_LightBase*)Data)->IsDelta;
	}

	AABB getBox(float eps) const
	{
		CALL_FUNC5(e_PointLight,e_DiffuseLight,e_DistantLight,e_SpotLight,e_InfiniteLight, getBox(eps))
		return AABB::Identity();
	}

	STD_VIRTUAL_SET
};

