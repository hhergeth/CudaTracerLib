#pragma once

#include <MathTypes.h>

CUDA_FUNC_IN float NoiseWeight(float t)
{
    float t3 = t*t*t;
    float t4 = t3*t;
    return 6.f*t4*t - 15.f*t4 + 10.f*t3;
}

CUDA_FUNC_IN float SmoothStep(float min, float max, float value)
{
    float v = clamp((value - min) / (max - min), 0.f, 1.f);
    return v * v * (-2.f * v  + 3.f);
}

CUDA_DEVICE CUDA_HOST float Grad(int x, int y, int z, float dx, float dy, float dz);

CUDA_DEVICE CUDA_HOST float Noise(float x, float y = .5f, float z = .5f);

CUDA_FUNC_IN float Noise(const float3 &P)
{
	return Noise(P.x, P.y, P.z);
}

CUDA_DEVICE CUDA_HOST float FBm(const float3 &P, const float3 &dpdx, const float3 &dpdy, float omega, int maxOctaves);

CUDA_DEVICE CUDA_HOST float Turbulence(const float3 &P, const float3 &dpdx, const float3 &dpdy, float omega, int maxOctaves);

CUDA_DEVICE CUDA_HOST float Lanczos(float x, float tau);

struct MapParameters
{
	float3 P;
	Frame sys;
	float2 uv;
	CUDA_FUNC_IN MapParameters(): sys(Frame()), uv(float2()), P(float3()){}
	CUDA_FUNC_IN MapParameters(const float3& p, const float2& u, const Frame& s)
		: sys(s), uv(u), P(p)
	{
	}
};

#define e_KernelUVMapping2D_TYPE 1
struct e_KernelUVMapping2D
{
	e_KernelUVMapping2D(float su = 1, float sv = 1, float du = 0, float dv = 0)
	{
		this->su = su;
		this->sv = sv;
		this->du = du;
		this->dv = dv;
	}
	CUDA_FUNC_IN void Map(const MapParameters &dg, float *s, float *t) const
	{
		*s = su * dg.uv.x + du;
		*t = sv * dg.uv.y + dv;
	}
	TYPE_FUNC(e_KernelUVMapping2D)
private:
    float su, sv, du, dv;
};

#define e_KernelSphericalMapping2D_TYPE 2
struct e_KernelSphericalMapping2D
{
	e_KernelSphericalMapping2D(float4x4& toSph)
		: WorldToTexture(toSph)
	{
	}
	CUDA_FUNC_IN void Map(const MapParameters &dg, float *s, float *t) const
	{
		sphere(dg.P, s, t);
	}
	TYPE_FUNC(e_KernelSphericalMapping2D)
private:
	CUDA_FUNC_IN void sphere(const float3 &P, float *s, float *t) const
	{
		float3 vec = normalize((WorldToTexture * P));
		float theta = MonteCarlo::SphericalTheta(vec);
		float phi = MonteCarlo::SphericalPhi(vec);
		*s = theta * INV_PI;
		*t = phi * INV_TWOPI;
	}
	float4x4 WorldToTexture;
};

#define e_KernelPlanarMapping2D_TYPE 3
struct e_KernelPlanarMapping2D
{
	e_KernelPlanarMapping2D(const float3 &vv1, const float3 &vv2, float dds = 0, float ddt = 0)
        : vs(vv1), vt(vv2), ds(dds), dt(ddt)
	{
	}
	CUDA_FUNC_IN void Map(const MapParameters &dg, float *s, float *t) const
	{
		float3 vec = dg.P;
		*s = ds + dot(vec, vs);
		*t = dt + dot(vec, vt);
	}
	TYPE_FUNC(e_KernelPlanarMapping2D)
private:
	const float3 vs, vt;
    const float ds, dt;
};

#define e_KernelCylindricalMapping2D_TYPE 4
struct e_KernelCylindricalMapping2D
{
	e_KernelCylindricalMapping2D(const float4x4 &toCyl)
        : WorldToTexture(toCyl)
	{
    }
	CUDA_FUNC_IN void Map(const MapParameters &dg, float *s, float *t) const
	{
		 cylinder(dg.P, s, t);
	}
	TYPE_FUNC(e_KernelCylindricalMapping2D)
private:
	CUDA_FUNC_IN void cylinder(const float3 &p, float *s, float *t) const
	{
        float3 vec = normalize(WorldToTexture * p);
        *s = (PI + atan2f(vec.y, vec.x)) / (2.f * PI);
        *t = vec.z;
    }
	float4x4 WorldToTexture;
};

#define MAP2D_SIZE RND_16(DMAX4(sizeof(e_KernelUVMapping2D), sizeof(e_KernelSphericalMapping2D), sizeof(e_KernelPlanarMapping2D), sizeof(e_KernelCylindricalMapping2D)))

struct e_KernelTextureMapping2D
{
private:
	unsigned char Data[MAP2D_SIZE];
	unsigned int type;
#define CALL_TYPE(t,f,r) \
	case t##_TYPE : \
		r ((t*)Data)->f; \
		break;
#define CALL_FUNC(r,f) \
	switch (type) \
	{ \
		CALL_TYPE(e_KernelUVMapping2D, f, r) \
		CALL_TYPE(e_KernelSphericalMapping2D, f, r) \
		CALL_TYPE(e_KernelPlanarMapping2D, f, r) \
		CALL_TYPE(e_KernelCylindricalMapping2D, f, r) \
	}
public:
	e_KernelTextureMapping2D()
	{
		type = 0;
	}
	template<typename T> void SetData(const T& val)
	{
		memcpy(Data, &val, sizeof(T));
		type = T::TYPE();
	}
	template<typename T> T* As()
	{
		return (T*)Data;
	}
	CUDA_FUNC_IN void Map(const MapParameters &dg, float *s, float *t) const
	{
		CALL_FUNC(, Map(dg, s, t))
	}
#undef CALL_TYPE
#undef CALL_FUNC
};

#define e_KernelIdentityMapping3D_TYPE 1
struct e_KernelIdentityMapping3D
{
	e_KernelIdentityMapping3D(const float4x4& x)
		: WorldToTexture(x)
	{
	}
	CUDA_FUNC_IN float3 Map(const MapParameters &dg) const
	{
		return WorldToTexture * dg.P;
	}
	TYPE_FUNC(e_KernelIdentityMapping3D)
private:
    float4x4 WorldToTexture;
};

#define MAP3D_SIZE RND_16(sizeof(e_KernelIdentityMapping3D))

struct e_KernelTextureMapping3D
{
private:
	unsigned char Data[MAP3D_SIZE];
	unsigned int type;
#define CALL_TYPE(t,f,r) \
	case t##_TYPE : \
		r ((t*)Data)->f; \
		break;
#define CALL_FUNC(r,f) \
	switch (type) \
	{ \
		CALL_TYPE(e_KernelIdentityMapping3D, f, r) \
	}
public:
	e_KernelTextureMapping3D()
	{
		type = 0;
	}
	template<typename T> void SetData(const T& val)
	{
		memcpy(Data, &val, sizeof(T));
		type = T::TYPE();
	}
	template<typename T> T* As()
	{
		return (T*)Data;
	}
	CUDA_FUNC_IN float3 Map(const MapParameters &dg) const
	{
		CALL_FUNC(return, Map(dg))
	}
#undef CALL_TYPE
#undef CALL_FUNC
};

template<typename T> static inline e_KernelTextureMapping2D CreateTextureMapping2D(T& val)
{
	e_KernelTextureMapping2D r;
	r.SetData(val);
	return r;
}

template<typename T> static inline e_KernelTextureMapping3D CreateTextureMapping3D(T& val)
{
	e_KernelTextureMapping3D r;
	r.SetData(val);
	return r;
}