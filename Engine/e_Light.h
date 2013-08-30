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
    float3 Intensity;
	float radius;

	e_PointLight(float3 p, float3 L, float r = 0)
		: e_LightBase(true), lightPos(p), Intensity(L), radius(r)
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
		float3 p,d;
		if(radius != 0)
		{
			float3 n = UniformSampleSphere(u1, u2) * radius;
			p = lightPos + n;
			//d = SampleCosineHemisphere(n, u1, u2);
			d = UniformSampleSphere(ls.uPos[0], ls.uPos[1]);
		}
		else
		{
			p = lightPos;
			d = UniformSampleSphere(ls.uPos[0], ls.uPos[1]);
		}	
		*ray = Ray(p, d);
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
		return AABB(lightPos - make_float3(MAX(eps, radius)), lightPos + make_float3(MAX(eps, radius)));
	}
	
	TYPE_FUNC(e_PointLight)
};

#define e_DiffuseLight_TYPE 2
struct e_DiffuseLight : public e_LightBase
{
	CUDA_ALIGN(16) float3 Lemit;
    CUDA_ALIGN(16) ShapeSet shapeSet;

	e_DiffuseLight(float3 L, ShapeSet& s)
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
	
	TYPE_FUNC(e_DiffuseLight)
};

#define e_DistantLight_TYPE 3
struct e_DistantLight : public e_LightBase
{
	float3 lightDir;
    float3 _L;
	Frame sys;

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
		float3 Pdisk = worldCenter + worldRadius * (d1 * sys.s + d2 * sys.t);
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
	
	TYPE_FUNC(e_DistantLight)
};

#define e_SpotLight_TYPE 4
struct e_SpotLight : public e_LightBase
{
    float3 lightPos;
    float3 Intensity;
    float cosTotalWidth, cosFalloffStart;
	Frame sys;

	e_SpotLight(float3 p, float3 t, float3 L, float width, float fall)
		: e_LightBase(true), lightPos(p), Intensity(L)
	{
		cosTotalWidth = cosf(Radians(width));
		cosFalloffStart = cosf(Radians(fall));
		sys = Frame(t - p);
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
		*ray = Ray(lightPos, sys.toWorld(v));
		*Ns = ray->direction;
		*pdf = UniformConePdf(cosTotalWidth);
		return Intensity * Falloff(v);
	}

	CUDA_FUNC_IN float3 Sample_L(const e_KernelDynamicScene& scene, const float3& p, const LightSample& ls, float* pdf, e_VisibilitySegment* seg) const
	{
		seg->SetSegment(p, 0, lightPos, 0);
		*pdf = 1.0f;
		return Intensity * Falloff(sys.toLocal(-seg->r.direction)) / DistanceSquared(lightPos, p);
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
	
	TYPE_FUNC(e_SpotLight)
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
};

#define e_InfiniteLight_TYPE 5
struct e_InfiniteLight : public e_LightBase
{
	//e_BufferReference<Distribution2D<4096, 4096>, Distribution2D<4096, 4096>> distribution;
	e_KernelMIPMap radianceMap;
	Distribution2D<4096, 4096>* pDist;

	e_InfiniteLight(const float3& power, e_BufferReference<Distribution2D<4096, 4096>, Distribution2D<4096, 4096>>& d, e_BufferReference<e_MIPMap, e_KernelMIPMap>& mip)
		: e_LightBase(false), pDist(0)
	{
		radianceMap = mip->getKernelData();
		void* pD = radianceMap.m_pDeviceData, *pH = malloc(mip->getBufferSize());
		cudaMemcpy(pH, pD, mip->getBufferSize(), cudaMemcpyDeviceToHost);
		radianceMap.m_pDeviceData = pH;
		unsigned int width = radianceMap.m_uWidth, height = radianceMap.m_uHeight;
		float filter = 1.0f / (float)MAX(width, height);
		float *img = new float[width*height];//I HATE new
		for (int v = 0; v < height; ++v)
		{
			float vp = (float)v / (float)height;
			float sinTheta = sinf(PI * float(v+.5f)/float(height));
			for (int u = 0; u < width; ++u)
			{
				float up = (float)u / (float)width;
				img[u+v*width] = y(radianceMap.Sample<float3>(make_float2(up, vp), filter));
				img[u+v*width] *= sinTheta;
			}
		}
		pDist = d.getDevice();
		Distribution2D<4096, 4096>* t = d.operator()();
		t->Initialize(img, width, height);
		delete [] img;
		d.Invalidate();
		free(pH);
		radianceMap.m_pDeviceData = pD;
	}

	CUDA_FUNC_IN float3 Power(const e_KernelDynamicScene& scene) const
	{
		float r = length(scene.m_sBox.Size()) / 2.0f;
		return PI * r * r * radianceMap.Sample<float3>(make_float2(0.5f, 0.5f), 0.5f);
	}

	CUDA_FUNC_IN float Pdf(const e_KernelDynamicScene& scene, const float3 &p, const float3 &wi) const
	{
		float theta = SphericalTheta(wi), phi = SphericalPhi(wi);
		float sintheta = sinf(theta);
		if (sintheta == 0.f)
			return 0.f;
		float p2 = pDist->Pdf(phi * INV_TWOPI, theta * INV_PI) / (2.f * PI * PI * sintheta);
		return p2;
	}

	CUDA_FUNC_IN float3 Sample_L(const e_KernelDynamicScene& scene, const LightSample &ls, float u1, float u2, Ray *ray, float3 *Ns, float *pdf) const
	{
		float uv[2], mapPdf;
		pDist->SampleContinuous(ls.uPos[0], ls.uPos[1], uv, &mapPdf);
		if (mapPdf == 0.f)
			return make_float3(0.f);
		float theta = uv[1] * PI, phi = uv[0] * 2.f * PI;
		float costheta = cosf(theta), sintheta = sinf(theta);
		float sinphi = sinf(phi), cosphi = cosf(phi);
		float3 d = -make_float3(sintheta * cosphi, sintheta * sinphi, costheta);
		*Ns = d;
		float3 worldCenter = scene.m_sBox.Center();
		float worldRadius = length(scene.m_sBox.Size()) / 2.0f;
		Frame sys(d);
		float d1, d2;
		ConcentricSampleDisk(u1, u2, &d1, &d2);
		float3 Pdisk = worldCenter + worldRadius * (d1 * sys.t + d2 * sys.s);
		*ray = Ray(Pdisk + worldRadius * -d, d);
		float directionPdf = mapPdf / (2.f * PI * PI * sintheta);
		float areaPdf = 1.f / (PI * worldRadius * worldRadius);
		*pdf = directionPdf * areaPdf;
		if (sintheta == 0.f)
			*pdf = 0.f;
		return radianceMap.Sample<float3>(make_float2(uv[0], uv[1]), 0);
	}

	CUDA_FUNC_IN float3 Sample_L(const e_KernelDynamicScene& scene, const float3& p, const LightSample& ls, float* pdf, e_VisibilitySegment* seg) const
	{
		float uv[2], mapPdf;
		pDist->SampleContinuous(ls.uPos[0], ls.uPos[1], uv, &mapPdf);
		if (mapPdf == 0.f)
			return make_float3(0.f);
		float theta = uv[1] * PI, phi = uv[0] * 2.f * PI;
		float costheta = cosf(theta), sintheta = sinf(theta);
		float sinphi = sinf(phi), cosphi = cosf(phi);
		float3 wi = make_float3(sintheta * cosphi, sintheta * sinphi, costheta);
		*pdf = mapPdf / (2.f * PI * PI * sintheta);
		if (sintheta == 0.f)
			*pdf = 0.f;
		seg->SetRay(p, 0, wi);
		return radianceMap.Sample<float3>(make_float2(uv[0], uv[1]), 0);
	}
	
	CUDA_FUNC_IN float3 L(const float3 &p, const float3 &n, const float3 &w) const
	{
		return make_float3(0);
	}

	CUDA_FUNC_IN float3 Le(const e_KernelDynamicScene& scene, const Ray& r) const
	{
		float s = SphericalPhi(r.direction) * INV_TWOPI, t = SphericalTheta(r.direction) * INV_PI;
		return radianceMap.Sample<float3>(make_float2(s, t), 0);
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

