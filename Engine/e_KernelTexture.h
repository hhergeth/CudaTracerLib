#pragma once

#include "e_KernelMapping.h"
#include "e_FileTexture.h"

typedef char e_String[256];

template <typename T, int SIZE> struct e_KernelTextureGE;

#define e_KernelBiLerpTexture_TYPE 1
template <typename T> struct e_KernelBiLerpTexture
{
	e_KernelBiLerpTexture(const e_KernelTextureMapping2D& m, const T &t00, const T &t01, const T &t10, const T &t11)
	{
		mapping = m;
		v00 = t00;
		v01 = t01;
		v10 = t10;
		v11 = t11;
	}
	CUDA_FUNC_IN T Evaluate(const MapParameters & dg) const
	{
		float2 uv = make_float2(0);
		mapping.Map(dg, &uv.x, &uv.y);
		return (1-uv.x)*(1-uv.y) * v00 + (1-uv.x)*(  uv.y) * v01 +
               (  uv.x)*(1-uv.y) * v10 + (  uv.x)*(  uv.y) * v11;
	}
	template<typename L> void LoadTextures(L callback)
	{
	}
	TYPE_FUNC(e_KernelBiLerpTexture)
private:
	e_KernelTextureMapping2D mapping;
	T v00, v01, v10, v11;
};

#define e_KernelConstantTexture_TYPE 3
template <typename T> struct e_KernelConstantTexture
{
	e_KernelConstantTexture(const T& v)
		: val(v)
	{

	}
	CUDA_FUNC_IN T Evaluate(const MapParameters & dg) const
	{
		return val;
	}
	template<typename L> void LoadTextures(L callback)
	{

	}
	TYPE_FUNC(e_KernelConstantTexture)
private:
	T val;
};

#define e_KernelFbmTexture_TYPE 5
template <typename T> struct e_KernelFbmTexture
{
	e_KernelFbmTexture(const e_KernelTextureMapping3D& m, int oct, float roughness, const T& v)
		: mapping(m), omega(roughness), octaves(oct), val(v)
	{

	}
	CUDA_FUNC_IN T Evaluate(const MapParameters & dg) const
	{
		float3 P = mapping.Map(dg);
		return val * FBm(P, make_float3(1, 0, 0), make_float3(0, 0, 1), omega, octaves);//BUG, need diffs...
	}
	template<typename L> void LoadTextures(L callback)
	{

	}
	TYPE_FUNC(e_KernelFbmTexture)
private:
	T val;
	float omega;
    int octaves;
	e_KernelTextureMapping3D mapping;
};

#define e_KernelImageTexture_TYPE 6
template <typename T> struct e_KernelImageTexture
{
	e_KernelImageTexture(const e_KernelTextureMapping2D& m, const char* _file)
		: mapping(m)
	{
		memcpy(file, _file, strlen(_file) + 1);
	}
	CUDA_FUNC_IN T Evaluate(const MapParameters & dg) const
	{
		float2 uv = make_float2(0);
		mapping.Map(dg, &uv.x, &uv.y);
#ifdef __CUDACC__
		return deviceTex->Sample<T>(uv);
#else
		return hostTex->Sample<T>(uv);
#endif
	}
	template<typename L> void LoadTextures(L callback)
	{
		deviceTex = callback(file).getDevice();
		hostTex = callback(file).getDeviceMapped();
	}
	TYPE_FUNC(e_KernelImageTexture)
public:
	e_KernelFileTexture* deviceTex;
	e_KernelFileTexture* hostTex;
	e_KernelTextureMapping2D mapping;
	e_String file;
};

#define e_KernelMarbleTexture_TYPE 7
struct e_KernelMarbleTexture
{
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
	template<typename L> void LoadTextures(L callback)
	{

	}
	TYPE_FUNC(e_KernelMarbleTexture)
private:
	int octaves;
    float omega, scale, variation;
	e_KernelTextureMapping3D mapping;
};

#define e_KernelUVTexture_TYPE 10
struct e_KernelUVTexture
{
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
	template<typename L> void LoadTextures(L callback)
	{
	}
	TYPE_FUNC(e_KernelUVTexture)
private:
	e_KernelTextureMapping2D mapping;
};

#define e_KernelWindyTexture_TYPE 11
template <typename T> struct e_KernelWindyTexture
{
	e_KernelWindyTexture(const e_KernelTextureMapping3D& m, const T& v)
		: mapping(m), val(v)
	{

	}
	CUDA_FUNC_IN T Evaluate(const MapParameters & dg) const
	{
		float3 P = mapping.Map(dg);
		float windStrength = FBm(.1f * P, .1f * make_float3(1, 0, 0), .1f * make_float3(0, 0, 1), .5f, 3);
        float waveHeight = FBm(P, make_float3(1, 0, 0), make_float3(0, 0, 1), .5f, 6);//BUG
        return val * fabsf(windStrength) * waveHeight;
	}
	template<typename L> void LoadTextures(L callback)
	{

	}
	TYPE_FUNC(e_KernelWindyTexture)
private:
	T val;
	e_KernelTextureMapping3D mapping;
};

#define e_KernelWrinkledTexture_TYPE 12
template <typename T> struct e_KernelWrinkledTexture
{
	e_KernelWrinkledTexture(const e_KernelTextureMapping3D& m, int oct, float roughness, const T& v)
		: mapping(m), omega(roughness), octaves(oct), val(v)
	{

	}
	CUDA_FUNC_IN T Evaluate(const MapParameters & dg) const
	{
		float3 P = mapping.Map(dg);
		return val * Turbulence(P, make_float3(1, 0, 0), make_float3(0, 0, 1), omega, octaves);//BUG, need diffs...
	}
	template<typename L> void LoadTextures(L callback)
	{

	}
	TYPE_FUNC(e_KernelWrinkledTexture)
private:
	T val;
	float omega;
    int octaves;
	e_KernelTextureMapping3D mapping;
};

#define CALL_FUNC(AB,r,f) \
	switch (type) \
	{ \
	case e_KernelBiLerpTexture_TYPE : \
		r ((e_KernelBiLerpTexture<AB>*)Data)->f; \
		break; \
	case e_KernelConstantTexture_TYPE : \
		r ((e_KernelConstantTexture<AB>*)Data)->f; \
		break; \
	case e_KernelFbmTexture_TYPE : \
		r ((e_KernelFbmTexture<AB>*)Data)->f; \
		break; \
	case e_KernelImageTexture_TYPE : \
		r ((e_KernelImageTexture<AB>*)Data)->f; \
		break; \
	case e_KernelMarbleTexture_TYPE : \
		r ((e_KernelMarbleTexture*)Data)->f; \
		break; \
	case e_KernelUVTexture_TYPE : \
		r ((e_KernelUVTexture*)Data)->f; \
		break; \
	case e_KernelWindyTexture_TYPE : \
		r ((e_KernelWindyTexture<AB>*)Data)->f; \
		break; \
	case e_KernelWrinkledTexture_TYPE : \
		r ((e_KernelWrinkledTexture<AB>*)Data)->f; \
		break; \
	}

#define CALL_FUNCO(AB,r,f) \
	switch (type) \
	{ \
	case e_KernelBiLerpTexture_TYPE : \
		r ((e_KernelBiLerpTexture<AB>*)Data)->f; \
		break; \
	case e_KernelConstantTexture_TYPE : \
		r ((e_KernelConstantTexture<AB>*)Data)->f; \
		break; \
	case e_KernelFbmTexture_TYPE : \
		r ((e_KernelFbmTexture<AB>*)Data)->f; \
		break; \
	case e_KernelImageTexture_TYPE : \
		r ((e_KernelImageTexture<AB>*)Data)->f; \
		break; \
	case e_KernelWindyTexture_TYPE : \
		r ((e_KernelWindyTexture<AB>*)Data)->f; \
		break; \
	case e_KernelWrinkledTexture_TYPE : \
		r ((e_KernelWrinkledTexture<AB>*)Data)->f; \
		break; \
	}

template <typename T, int SIZE> struct CUDA_ALIGN(16) e_KernelTextureGE
{
public:
	CUDA_ALIGN(16) unsigned char Data[SIZE];
	unsigned int type;
public:
#ifdef __CUDACC__
	CUDA_DEVICE e_KernelTextureGE()
	{
	}
#else
	CUDA_HOST e_KernelTextureGE()
	{
		type = 0;
	}
#endif
	template<typename T> void SetData(const T& val)
	{
		memcpy(Data, &val, sizeof(T));
		type = T::TYPE();
	}
	template<typename T> T* As()
	{
		return (T*)Data;
	}
	CUDA_FUNC_IN T Evaluate(const MapParameters & dg) const
	{
		CALL_FUNCO(T, return, Evaluate(dg))
	}
	template<typename L> void LoadTextures(L callback)
	{
		CALL_FUNC(T, , LoadTextures(callback))
	}
};

template <int SIZE> struct CUDA_ALIGN(16) e_KernelTextureGE<Spectrum, SIZE>
{
public:
	CUDA_ALIGN(16) unsigned char Data[SIZE];
	unsigned int type;
public:
#ifdef __CUDACC__
	CUDA_DEVICE e_KernelTextureGE()
	{
	}
#else
	CUDA_HOST e_KernelTextureGE()
	{
		type = 0;
	}
#endif
	template<typename T> void SetData(const T& val)
	{
		memcpy(Data, &val, sizeof(T));
		type = T::TYPE();
	}
	template<typename T> T* As()
	{
		return (T*)Data;
	}
	CUDA_FUNC_IN Spectrum Evaluate(const MapParameters & dg) const
	{
		CALL_FUNC(Spectrum, return, Evaluate(dg))
	}
	template<typename L> void LoadTextures(L callback)
	{
		CALL_FUNC(Spectrum, , LoadTextures(callback))
	}
};

#undef CALL_TYPE
#undef CALL_FUNC
#undef CALL_FUNCO

#define LType float4x4
#define STD_TEX_SIZE (RND_16(DMAX8(sizeof(e_KernelBiLerpTexture<LType>), sizeof(e_KernelConstantTexture<LType>), sizeof(e_KernelFbmTexture<LType>), sizeof(e_KernelImageTexture<LType>), \
								  sizeof(e_KernelMarbleTexture), sizeof(e_KernelUVTexture), sizeof(e_KernelWindyTexture<LType>), sizeof(e_KernelWrinkledTexture<LType>))) + 12)

template <typename T> struct e_KernelTexture : public e_KernelTextureGE<T, STD_TEX_SIZE>
{

};

template<typename V, template<typename> class U> static inline e_KernelTexture<V> CreateTexture(const U<V>& val)
{
	e_KernelTexture<V> r;
	r.SetData(val);
	return r;
}

template<typename T> e_KernelTexture<T> CreateTexture(const char* p, const T& col)
{
	e_KernelTexture<T> r;
	if(p && strlen(p))
	{
		char* c = new char[strlen(p) + 1];
		ZeroMemory(c, strlen(p) + 1);
		memcpy(c, p, strlen(p));
		r.SetData(e_KernelImageTexture<T>(CreateTextureMapping2D(e_KernelUVMapping2D()), c));
	}
	else r.SetData(e_KernelConstantTexture<T>(col));
	return r;
}