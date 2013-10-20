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

struct e_TriangleData;
struct MapParameters
{
	float3 P;
	Frame sys;
	float2 uv;
	float2 bary;
	const e_TriangleData* Shape;
	CUDA_FUNC_IN MapParameters(){}
	//CUDA_FUNC_IN MapParameters(): sys(Frame()), uv(float2()), P(float3()), bary(float2()){}
	CUDA_FUNC_IN MapParameters(const float3& p, const float2& u, const Frame& s, const float2& b, const e_TriangleData* S)
		: sys(s), uv(u), P(p), bary(b), Shape(S)
	{
	}
};

struct e_KernelMappingBase : public e_BaseType
{
};

#define e_KernelUVMapping2D_TYPE 1
struct e_KernelUVMapping2D : public e_KernelMappingBase
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
    float su, sv, du, dv;
};

#define e_KernelSphericalMapping2D_TYPE 2
struct e_KernelSphericalMapping2D : public e_KernelMappingBase
{
	e_KernelSphericalMapping2D(){}
	e_KernelSphericalMapping2D(float4x4& toSph)
		: WorldToTexture(toSph)
	{
	}
	CUDA_FUNC_IN void Map(const MapParameters &dg, float *s, float *t) const
	{
		sphere(dg.P, s, t);
	}
	TYPE_FUNC(e_KernelSphericalMapping2D)
	float4x4 WorldToTexture;
private:
	CUDA_FUNC_IN void sphere(const float3 &P, float *s, float *t) const
	{
		float3 vec = normalize((WorldToTexture * P));
		float theta = MonteCarlo::SphericalTheta(vec);
		float phi = MonteCarlo::SphericalPhi(vec);
		*s = theta * INV_PI;
		*t = phi * INV_TWOPI;
	}
};

#define e_KernelPlanarMapping2D_TYPE 3
struct e_KernelPlanarMapping2D : public e_KernelMappingBase
{
	e_KernelPlanarMapping2D(){}
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
	float3 vs, vt;
    float ds, dt;
};

#define e_KernelCylindricalMapping2D_TYPE 4
struct e_KernelCylindricalMapping2D : public e_KernelMappingBase
{
	e_KernelCylindricalMapping2D(){}
	e_KernelCylindricalMapping2D(const float4x4 &toCyl)
        : WorldToTexture(toCyl)
	{
    }
	CUDA_FUNC_IN void Map(const MapParameters &dg, float *s, float *t) const
	{
		 cylinder(dg.P, s, t);
	}
	TYPE_FUNC(e_KernelCylindricalMapping2D)
	float4x4 WorldToTexture;
private:
	CUDA_FUNC_IN void cylinder(const float3 &p, float *s, float *t) const
	{
        float3 vec = normalize(WorldToTexture * p);
        *s = (PI + atan2f(vec.y, vec.x)) / (2.f * PI);
        *t = vec.z;
    }
};

#define MAP2D_SIZE RND_16(DMAX4(sizeof(e_KernelUVMapping2D), sizeof(e_KernelSphericalMapping2D), sizeof(e_KernelPlanarMapping2D), sizeof(e_KernelCylindricalMapping2D)))

struct e_KernelTextureMapping2D : public e_AggregateBaseType<e_KernelMappingBase, MAP2D_SIZE>
{
public:
	e_KernelTextureMapping2D()
	{
		type = 0;
	}
	CUDA_FUNC_IN void Map(const MapParameters &dg, float *s, float *t) const
	{
		CALL_FUNC4(e_KernelUVMapping2D,e_KernelSphericalMapping2D,e_KernelPlanarMapping2D,e_KernelCylindricalMapping2D, Map(dg, s, t))
	}
};

#define e_KernelIdentityMapping3D_TYPE 1
struct e_KernelIdentityMapping3D : public e_KernelMappingBase
{
	e_KernelIdentityMapping3D(){}
	e_KernelIdentityMapping3D(const float4x4& x)
		: WorldToTexture(x)
	{
	}
	CUDA_FUNC_IN float3 Map(const MapParameters &dg) const
	{
		return WorldToTexture * dg.P;
	}
	TYPE_FUNC(e_KernelIdentityMapping3D)
    float4x4 WorldToTexture;
};

#define MAP3D_SIZE RND_16(sizeof(e_KernelIdentityMapping3D))

struct e_KernelTextureMapping3D : public e_AggregateBaseType<e_KernelMappingBase, MAP3D_SIZE>
{
public:
	e_KernelTextureMapping3D()
	{
		type = 0;
	}
	CUDA_FUNC_IN float3 Map(const MapParameters &dg) const
	{
		CALL_FUNC1(e_KernelIdentityMapping3D, Map(dg))
		return make_float3(0,0,0);
	}
};

template<typename T> static e_KernelTextureMapping2D CreateTextureMapping2D(T& val)
{
	e_KernelTextureMapping2D r;
	r.SetData(val);
	return r;
}

template<typename T> static e_KernelTextureMapping3D CreateTextureMapping3D(T& val)
{
	e_KernelTextureMapping3D r;
	r.SetData(val);
	return r;
}