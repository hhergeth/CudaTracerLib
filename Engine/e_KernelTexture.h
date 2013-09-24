#pragma once

#include "e_KernelMapping.h"
#include "e_FileTexture.h"

typedef char e_String[256];

struct e_KernelTextureBase : public e_BaseType
{
};

#define e_KernelBiLerpTexture_TYPE 1
struct e_KernelBiLerpTexture : public e_KernelTextureBase
{
	e_KernelBiLerpTexture(){}
	e_KernelBiLerpTexture(const e_KernelTextureMapping2D& m, const Spectrum &t00, const Spectrum &t01, const Spectrum &t10, const Spectrum &t11)
	{
		mapping = m;
		v00 = t00;
		v01 = t01;
		v10 = t10;
		v11 = t11;
	}
	CUDA_FUNC_IN Spectrum Evaluate(const MapParameters & dg) const
	{
		float2 uv = make_float2(0);
		mapping.Map(dg, &uv.x, &uv.y);
		return (1-uv.x)*(1-uv.y) * v00 + (1-uv.x)*(  uv.y) * v01 +
               (  uv.x)*(1-uv.y) * v10 + (  uv.x)*(  uv.y) * v11;
	}
	CUDA_FUNC_IN Spectrum Average()
	{
		return (v00+v01+v10+v11) * 0.25f;
	}
	template<typename L> void LoadTextures(L callback)
	{
	}
	TYPE_FUNC(e_KernelBiLerpTexture)
	e_KernelTextureMapping2D mapping;
	Spectrum v00, v01, v10, v11;
};

#define e_KernelConstantTexture_TYPE 3
struct e_KernelConstantTexture : public e_KernelTextureBase
{
	e_KernelConstantTexture(){}
	e_KernelConstantTexture(const Spectrum& v)
		: val(v)
	{

	}
	CUDA_FUNC_IN Spectrum Evaluate(const MapParameters & dg) const
	{
		return val;
	}
	CUDA_FUNC_IN Spectrum Average()
	{
		return val;
	}
	template<typename L> void LoadTextures(L callback)
	{

	}
	TYPE_FUNC(e_KernelConstantTexture)
	Spectrum val;
};

#define e_KernelCheckerboardTexture_TYPE 4
struct e_KernelCheckerboardTexture : public e_KernelTextureBase
{
	e_KernelCheckerboardTexture(){}
	e_KernelCheckerboardTexture(const Spectrum& u, const Spectrum& v, const e_KernelTextureMapping2D& m)
		: val0(u), val1(v), mapping(m)
	{

	}
	CUDA_FUNC_IN Spectrum Evaluate(const MapParameters & dg) const
	{
		float2 uv = make_float2(0);
		mapping.Map(dg, &uv.x, &uv.y);
		int x = 2*math::modulo((int) (uv.x * 2), 2) - 1,
			y = 2*math::modulo((int) (uv.y * 2), 2) - 1;

		if (x*y == 1)
			return val0;
		else
			return val1;
	}
	CUDA_FUNC_IN Spectrum Average()
	{
		return (val0 + val1) / 2.0f;
	}
	template<typename L> void LoadTextures(L callback)
	{

	}
	TYPE_FUNC(e_KernelCheckerboardTexture)
	Spectrum val0, val1;
	e_KernelTextureMapping2D mapping;
};

#define e_KernelFbmTexture_TYPE 5
struct e_KernelFbmTexture : public e_KernelTextureBase
{
	e_KernelFbmTexture(){}
	e_KernelFbmTexture(const e_KernelTextureMapping3D& m, int oct, float roughness, const Spectrum& v)
		: mapping(m), omega(roughness), octaves(oct), val(v)
	{

	}
	CUDA_FUNC_IN Spectrum Evaluate(const MapParameters & dg) const
	{
		float3 P = mapping.Map(dg);
		return val * FBm(P, make_float3(1, 0, 0), make_float3(0, 0, 1), omega, octaves);//BUG, need diffs...
	}
	CUDA_FUNC_IN Spectrum Average()
	{
		return val;
	}
	template<typename L> void LoadTextures(L callback)
	{

	}
	TYPE_FUNC(e_KernelFbmTexture)
	Spectrum val;
	float omega;
    int octaves;
	e_KernelTextureMapping3D mapping;
};

#define e_KernelImageTexture_TYPE 6
struct e_KernelImageTexture : public e_KernelTextureBase
{
	e_KernelImageTexture(){}
	e_KernelImageTexture(const e_KernelTextureMapping2D& m, const char* _file)
		: mapping(m)
	{
		memcpy(file, _file, strlen(_file) + 1);
	}
	CUDA_FUNC_IN Spectrum Evaluate(const MapParameters & dg) const
	{
		float2 uv = make_float2(0);
		mapping.Map(dg, &uv.x, &uv.y);
#ifdef __CUDACC__
		return deviceTex->Sample(uv);
#else
		return hostTex->Sample(uv);
#endif
	}
	CUDA_FUNC_IN Spectrum Average()
	{
#ifdef __CUDACC__
		return deviceTex->Sample(make_float2(0));
#else
		return hostTex->Sample(make_float2(0));
#endif
	}
	template<typename L> void LoadTextures(L callback)
	{
		deviceTex = callback(file, false).getDevice();
		hostTex = callback(file, false).getDeviceMapped();
	}
	TYPE_FUNC(e_KernelImageTexture)
	e_KernelMIPMap* deviceTex;
	e_KernelMIPMap* hostTex;
	e_KernelTextureMapping2D mapping;
	e_String file;
};

#define e_KernelMarbleTexture_TYPE 7
struct e_KernelMarbleTexture : public e_KernelTextureBase
{
	e_KernelMarbleTexture(){}
	e_KernelMarbleTexture(const e_KernelTextureMapping3D& m, int oct, float roughness, float sc, float var)
		: mapping(m), omega(roughness), octaves(oct), scale(sc), variation(var)
	{

	}
	CUDA_FUNC_IN Spectrum Evaluate(const MapParameters & dg) const
	{
		float3 P = mapping.Map(dg);
		P *= scale;
        float marble = P.y + variation * FBm(P, scale * make_float3(1, 0, 0), scale * make_float3(0, 0, 1), omega, octaves);//BUG, need diffs...
        float t = .5f + .5f * sinf(marble);
        // Evaluate marble spline at _t_
        Spectrum c[] = { Spectrum(.58f, .58f, .6f),	 Spectrum( .58f, .58f, .6f ), Spectrum( .58f, .58f, .6f ),
						 Spectrum( .5f, .5f, .5f ),	 Spectrum( .6f, .59f, .58f ), Spectrum( .58f, .58f, .6f ),
						 Spectrum( .58f, .58f, .6f ), Spectrum(.2f, .2f, .33f ),	 Spectrum( .58f, .58f, .6f ), };
#define NC  sizeof(c) / sizeof(c[0])
#define NSEG (NC-3)
        int first = Floor2Int(t * NSEG);
        t = (t * NSEG - first);
        Spectrum c0 = c[first];
        Spectrum c1 = c[first+1];
        Spectrum c2 = c[first+2];
        Spectrum c3 = c[first+3];
        // Bezier spline evaluated with de Castilejau's algorithm
        Spectrum s0 = (1.f - t) * c0 + t * c1;
        Spectrum s1 = (1.f - t) * c1 + t * c2;
        Spectrum s2 = (1.f - t) * c2 + t * c3;
        s0 = (1.f - t) * s0 + t * s1;
        s1 = (1.f - t) * s1 + t * s2;
        // Extra scale of 1.5 to increase variation among colors
        return 1.5f * ((1.f - t) * s0 + t * s1);
#undef NC
#undef NSEG
	}
	CUDA_FUNC_IN Spectrum Average()
	{
		Spectrum c[] = { Spectrum(.58f, .58f, .6f),	 Spectrum( .58f, .58f, .6f ), Spectrum( .58f, .58f, .6f ),
						 Spectrum( .5f, .5f, .5f ),	 Spectrum( .6f, .59f, .58f ), Spectrum( .58f, .58f, .6f ),
						 Spectrum( .58f, .58f, .6f ), Spectrum(.2f, .2f, .33f ),	 Spectrum( .58f, .58f, .6f ), };
#define NC  sizeof(c) / sizeof(c[0])
		Spectrum r(0.0f);
		for(int i = 0; i < NC; i++)
			r += c[i];
		return r / float(NC);
	}
	template<typename L> void LoadTextures(L callback)
	{

	}
	TYPE_FUNC(e_KernelMarbleTexture)
	int octaves;
    float omega, scale, variation;
	e_KernelTextureMapping3D mapping;
};

#define e_KernelUVTexture_TYPE 10
struct e_KernelUVTexture : public e_KernelTextureBase
{
	e_KernelUVTexture(){}
	e_KernelUVTexture(const e_KernelTextureMapping2D& m)
		: mapping(m)
	{
	}
	CUDA_FUNC_IN Spectrum Evaluate(const MapParameters & dg) const
	{
		float2 uv = make_float2(0);
		mapping.Map(dg, &uv.x, &uv.y);
		return Spectrum(frac(uv.x), frac(uv.y), 0);
	}
	CUDA_FUNC_IN Spectrum Average()
	{
		return Spectrum(0.5f);
	}
	template<typename L> void LoadTextures(L callback)
	{
	}
	TYPE_FUNC(e_KernelUVTexture)
	e_KernelTextureMapping2D mapping;
};

#define e_KernelWindyTexture_TYPE 11
struct e_KernelWindyTexture : public e_KernelTextureBase
{
	e_KernelWindyTexture(){}
	e_KernelWindyTexture(const e_KernelTextureMapping3D& m, const Spectrum& v)
		: mapping(m), val(v)
	{

	}
	CUDA_FUNC_IN Spectrum Evaluate(const MapParameters & dg) const
	{
		float3 P = mapping.Map(dg);
		float windStrength = FBm(.1f * P, .1f * make_float3(1, 0, 0), .1f * make_float3(0, 0, 1), .5f, 3);
        float waveHeight = FBm(P, make_float3(1, 0, 0), make_float3(0, 0, 1), .5f, 6);//BUG
        return val * fabsf(windStrength) * waveHeight;
	}
	CUDA_FUNC_IN Spectrum Average()
	{
		return val;
	}
	template<typename L> void LoadTextures(L callback)
	{

	}
	TYPE_FUNC(e_KernelWindyTexture)
	Spectrum val;
	e_KernelTextureMapping3D mapping;
};

#define e_KernelWrinkledTexture_TYPE 12
struct e_KernelWrinkledTexture : public e_KernelTextureBase
{
	e_KernelWrinkledTexture(){}
	e_KernelWrinkledTexture(const e_KernelTextureMapping3D& m, int oct, float roughness, const Spectrum& v)
		: mapping(m), omega(roughness), octaves(oct), val(v)
	{

	}
	CUDA_FUNC_IN Spectrum Evaluate(const MapParameters & dg) const
	{
		float3 P = mapping.Map(dg);
		return val * Turbulence(P, make_float3(1, 0, 0), make_float3(0, 0, 1), omega, octaves);//BUG, need diffs...
	}
	CUDA_FUNC_IN Spectrum Average()
	{
		return val;
	}
	template<typename L> void LoadTextures(L callback)
	{

	}
	TYPE_FUNC(e_KernelWrinkledTexture)
	Spectrum val;
	float omega;
    int octaves;
	e_KernelTextureMapping3D mapping;
};

#define e_KernelWireframeTexture_TYPE 13
struct e_KernelWireframeTexture : public e_KernelTextureBase
{
	e_KernelWireframeTexture(float lineWidth = 0.1f, const Spectrum& interior = Spectrum(0.5f), const Spectrum& edge = Spectrum(0.0f))
		: width(lineWidth), interiorColor(interior), edgeColor(edge)
	{
	}
	CUDA_FUNC_IN Spectrum Evaluate(const MapParameters & dg) const
	{
		if(dg.bary.x < width || dg.bary.y < width || dg.bary.x + dg.bary.y > 1.0f - width)
			return edgeColor;
		else return interiorColor;
	}
	CUDA_FUNC_IN Spectrum Average()
	{
		return Spectrum(0.1f);
	}
	template<typename L> void LoadTextures(L callback)
	{
	}
	TYPE_FUNC(e_KernelWireframeTexture)
	Spectrum interiorColor, edgeColor;
	float width;
};

#define TEX_SIZE (RND_16(DMAX9(sizeof(e_KernelBiLerpTexture), sizeof(e_KernelConstantTexture), sizeof(e_KernelFbmTexture), sizeof(e_KernelImageTexture), \
								  sizeof(e_KernelMarbleTexture), sizeof(e_KernelUVTexture), sizeof(e_KernelWindyTexture), sizeof(e_KernelWrinkledTexture), \
								  DMAX2(sizeof(e_KernelCheckerboardTexture), sizeof(e_KernelWireframeTexture)))) + 12)

struct CUDA_ALIGN(16) e_KernelTexture : public e_AggregateBaseType<e_KernelTextureBase, TEX_SIZE>
{
public:
#ifdef __CUDACC__
	CUDA_FUNC_IN e_KernelTexture()
	{
	}
#else
	CUDA_FUNC_IN e_KernelTexture()
	{
		type = 0;
	}
#endif
	CUDA_FUNC_IN Spectrum Evaluate(const MapParameters & dg) const
	{
		CALL_FUNC10(e_KernelBiLerpTexture,e_KernelConstantTexture,e_KernelCheckerboardTexture,e_KernelFbmTexture,e_KernelImageTexture,e_KernelMarbleTexture,e_KernelUVTexture,e_KernelWindyTexture,e_KernelWrinkledTexture,e_KernelWireframeTexture, Evaluate(dg))
		return Spectrum(0.0f);
	}
	CUDA_FUNC_IN Spectrum Average()
	{
		CALL_FUNC10(e_KernelBiLerpTexture,e_KernelConstantTexture,e_KernelCheckerboardTexture,e_KernelFbmTexture,e_KernelImageTexture,e_KernelMarbleTexture,e_KernelUVTexture,e_KernelWindyTexture,e_KernelWrinkledTexture,e_KernelWireframeTexture, Average())
		return Spectrum(0.0f);
	}
	template<typename L> void LoadTextures(L callback)
	{
		CALL_FUNC10(e_KernelBiLerpTexture,e_KernelConstantTexture,e_KernelCheckerboardTexture,e_KernelFbmTexture,e_KernelImageTexture,e_KernelMarbleTexture,e_KernelUVTexture,e_KernelWindyTexture,e_KernelWrinkledTexture,e_KernelWireframeTexture, LoadTextures(callback))
	}
};

template<typename U> static e_KernelTexture CreateTexture(const U& val)
{
	e_KernelTexture r;
	r.SetData(val);
	return r;
}

static e_KernelTexture CreateTexture(const char* p)
{
	e_KernelImageTexture f(CreateTextureMapping2D(e_KernelUVMapping2D()), p);
	return CreateTexture(f);
}

static e_KernelTexture CreateTexture(const Spectrum& col)
{
	e_KernelConstantTexture f(col);
	return CreateTexture(f);
}

static e_KernelTexture CreateTexture(const char* p, const Spectrum& col)
{
	if(p && *p)
		return CreateTexture(p);
	else return CreateTexture(col);
}