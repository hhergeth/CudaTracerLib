#pragma once

//do not worry :') missing template class sizes at run time is okay, cause memory layout will be correct!

#include "e_KernelMapping.h"
#include "e_FileTexture.h"

typedef char e_String[256];

#define STD_TEX_SIZE 2048

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

#define e_KernelCheckerboardTexture_TYPE 2
template <typename T, int USIZE, int VSIZE> struct e_KernelCheckerboardTexture
{
	e_KernelCheckerboardTexture(const e_KernelTextureMapping2D& m, const e_KernelTextureGE<T, USIZE>& tex1, const e_KernelTextureGE<T, VSIZE>& tex2)
		: mapping(m), texA(tex1), texB(tex2)
	{
		skipSize = sizeof(e_KernelTextureGE<T, USIZE>);
	}
	CUDA_FUNC_IN T Evaluate(const MapParameters & dg) const
	{
		e_KernelTextureGE<T, VSIZE>* B = (e_KernelTextureGE<T, VSIZE>*)(((unsigned int)&texA) + skipSize);
		float2 uv = make_float2(0);
		mapping.Map(dg, &uv.x, &uv.y);
		if ((Floor2Int(uv.x) + Floor2Int(uv.y)) % 2 == 0)
            return texA.Evaluate(dg);
        return B->Evaluate(dg);
	}
	template<typename L> void LoadTextures(L callback)
	{
		e_KernelTextureGE<T, VSIZE>* B = (e_KernelTextureGE<T, VSIZE>*)(((unsigned int)&texA) + skipSize);
		texA.LoadTextures(callback);
		B->LoadTextures(callback);
	}
	TYPE_FUNC(e_KernelCheckerboardTexture)
private:
	unsigned int skipSize;
	e_KernelTextureMapping2D mapping;
	e_KernelTextureGE<T, USIZE> texA;
	e_KernelTextureGE<T, VSIZE> texB;
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

#define e_KernelDotsTexture_TYPE 4
template <typename T, int USIZE, int VSIZE> struct e_KernelDotsTexture
{
	e_KernelDotsTexture(const e_KernelTextureMapping2D& m, const e_KernelTextureGE<T, USIZE>& _insideDot, const e_KernelTextureGE<T, VSIZE>& _outsideDot)
		: mapping(m), insideDot(_insideDot), outsideDot(_outsideDot)
	{
		skipSize = sizeof(e_KernelTextureGE<T, USIZE>);
	}
	CUDA_FUNC_IN T Evaluate(const MapParameters & dg) const
	{
		e_KernelTextureGE<T, VSIZE>* B = (e_KernelTextureGE<T, VSIZE>*)(((unsigned int)&insideDot) + skipSize);
		float2 uv = make_float2(0);
		mapping.Map(dg, &uv.x, &uv.y);
		int sCell = Floor2Int(uv.x + .5f), tCell = Floor2Int(uv.y + .5f);
		if (Noise(sCell+.5f, tCell+.5f) > 0)
		{
			float radius = .35f;
            float maxShift = 0.5f - radius;
            float sCenter = sCell + maxShift *
                Noise(sCell + 1.5f, tCell + 2.8f);
            float tCenter = tCell + maxShift *
                Noise(sCell + 4.5f, tCell + 9.8f);
			float ds = uv.x - sCenter, dt = uv.y - tCenter;
            if (ds*ds + dt*dt < radius*radius)
                return insideDot.Evaluate(dg);
		}
		else return B->Evaluate(dg);
	}
	template<typename L> void LoadTextures(L callback)
	{
		e_KernelTextureGE<T, VSIZE>* B = (e_KernelTextureGE<T, VSIZE>*)(((unsigned int)&insideDot) + skipSize);
		insideDot.LoadTextures(callback);
		B->LoadTextures(callback);
	}
	TYPE_FUNC(e_KernelDotsTexture)
private:
	unsigned int skipSize;
	e_KernelTextureMapping2D mapping;
	e_KernelTextureGE<T, USIZE> insideDot;
	e_KernelTextureGE<T, VSIZE> outsideDot;
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
		return tex.Sample<T>(uv);
	}
	template<typename L> void LoadTextures(L callback)
	{
		tex = callback(file)->getKernelData();
	}
	TYPE_FUNC(e_KernelImageTexture)
public:
	e_KernelTextureMapping2D mapping;
	e_String file;
	e_KernelFileTexture tex;
};

#define e_KernelMarbleTexture_TYPE 7
struct e_KernelMarbleTexture
{
	e_KernelMarbleTexture(const e_KernelTextureMapping3D& m, int oct, float roughness, float sc, float var)
		: mapping(m), omega(roughness), octaves(oct), scale(sc), variation(var)
	{

	}
	CUDA_FUNC_IN float3 Evaluate(const MapParameters & dg) const
	{
		float3 P = mapping.Map(dg);
		P *= scale;
        float marble = P.y + variation * FBm(P, scale * make_float3(1, 0, 0), scale * make_float3(0, 0, 1), omega, octaves);//BUG, need diffs...
        float t = .5f + .5f * sinf(marble);
        // Evaluate marble spline at _t_
        float3 c[] = { make_float3(.58f, .58f, .6f), make_float3( .58f, .58f, .6f ), make_float3( .58f, .58f, .6f ),
            make_float3( .5f, .5f, .5f ), make_float3( .6f, .59f, .58f ), make_float3( .58f, .58f, .6f ),
            make_float3( .58f, .58f, .6f ), make_float3(.2f, .2f, .33f ), make_float3( .58f, .58f, .6f ), };
#define NC  sizeof(c) / sizeof(c[0])
#define NSEG (NC-3)
        int first = Floor2Int(t * NSEG);
        t = (t * NSEG - first);
        float3 c0 = c[first];
        float3 c1 = c[first+1];
        float3 c2 = c[first+2];
        float3 c3 = c[first+3];
        // Bezier spline evaluated with de Castilejau's algorithm
        float3 s0 = (1.f - t) * c0 + t * c1;
        float3 s1 = (1.f - t) * c1 + t * c2;
        float3 s2 = (1.f - t) * c2 + t * c3;
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

#define e_KernelMixTexture_TYPE 8
template <typename T, int USIZE, int VSIZE, int WSIZE> struct e_KernelMixTexture
{
	e_KernelMixTexture(const e_KernelTextureGE<T, USIZE>& tex1, const e_KernelTextureGE<T, VSIZE>& tex2, const e_KernelTextureGE<float, WSIZE>& _amount)
		: texA(tex1), texB(tex2), amount(_amount)
	{
		skip1 = sizeof(e_KernelTextureGE<float, WSIZE>);
		skip2 = sizeof(e_KernelTextureGE<T, USIZE>);
	}
	CUDA_FUNC_IN T Evaluate(const MapParameters & dg) const
	{
		e_KernelTextureGE<T, USIZE>* A = (e_KernelTextureGE<T, USIZE>*)(((unsigned int)&amount) + skip1);
		e_KernelTextureGE<T, VSIZE>* B = (e_KernelTextureGE<T, VSIZE>*)(((unsigned int)&amount) + skip1 + skip2);
		T t1 = A->Evaluate(dg), t2 = B->Evaluate(dg);
        float amt = amount.Evaluate(dg);
        return (1.f - amt) * t1 + amt * t2;
	}
	template<typename L> void LoadTextures(L callback)
	{
		e_KernelTextureGE<T, USIZE>* A = (e_KernelTextureGE<T, USIZE>*)(((unsigned int)&amount) + skip1);
		e_KernelTextureGE<T, VSIZE>* B = (e_KernelTextureGE<T, VSIZE>*)(((unsigned int)&amount) + skip1 + skip2);
		A->LoadTextures(callback);
		B->LoadTextures(callback);
		amount.LoadTextures(callback);
	}
	TYPE_FUNC(e_KernelMixTexture)
private:
	unsigned int skip1, skip2;
	e_KernelTextureGE<float, WSIZE> amount;
	e_KernelTextureGE<T, USIZE> texA;
	e_KernelTextureGE<T, VSIZE> texB;
};
/*
#define e_KernelScaleTexture_TYPE 9
template <typename T1, typename T2, int USIZE, int VSIZE> struct e_KernelScaleTexture
{
	e_KernelScaleTexture(const e_KernelTextureGE<T1, USIZE>& tex1, const e_KernelTextureGE<T2, VSIZE>& tex2)
		: texA(tex1), texB(tex2)
	{
		skipSize = sizeof(e_KernelTextureGE<T1, USIZE>);
	}
	CUDA_FUNC_IN T2 Evaluate(const MapParameters & dg) const
	{
		e_KernelTextureGE<T, VSIZE>* B = (e_KernelTextureGE<T, VSIZE>*)(((unsigned int)&texA) + skipSize);
		return texA.Evaluate(dg) * B->Evaluate(dg);
	}
	template<typename L> void LoadTextures(L callback)
	{
		e_KernelTextureGE<T, VSIZE>* B = (e_KernelTextureGE<T, VSIZE>*)(((unsigned int)&texA) + skipSize);
		texA.LoadTextures(callback);
		B->LoadTextures(callback);
	}
	TYPE_FUNC(e_KernelScaleTexture)
private:
	unsigned int skipSize;
	e_KernelTextureGE<T1, USIZE> texA;
	e_KernelTextureGE<T2, VSIZE> texB;
};
*/
#define e_KernelUVTexture_TYPE 10
struct e_KernelUVTexture
{
	e_KernelUVTexture(const e_KernelTextureMapping2D& m)
		: mapping(m)
	{
	}
	CUDA_FUNC_IN float3 Evaluate(const MapParameters & dg) const
	{
		float2 uv = make_float2(0);
		mapping.Map(dg, &uv.x, &uv.y);
		return make_float3(frac(uv.x), frac(uv.y), 0);
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
	case e_KernelCheckerboardTexture_TYPE : \
		r ((e_KernelCheckerboardTexture<AB, STD_TEX_SIZE, STD_TEX_SIZE>*)Data)->f; \
		break; \
	case e_KernelConstantTexture_TYPE : \
		r ((e_KernelConstantTexture<AB>*)Data)->f; \
		break; \
	case e_KernelDotsTexture_TYPE : \
		r ((e_KernelDotsTexture<AB, STD_TEX_SIZE, STD_TEX_SIZE>*)Data)->f; \
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
	case e_KernelMixTexture_TYPE : \
		r ((e_KernelMixTexture<AB, STD_TEX_SIZE, STD_TEX_SIZE, STD_TEX_SIZE>*)Data)->f; \
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
	case e_KernelCheckerboardTexture_TYPE : \
		r ((e_KernelCheckerboardTexture<AB, STD_TEX_SIZE, STD_TEX_SIZE>*)Data)->f; \
		break; \
	case e_KernelConstantTexture_TYPE : \
		r ((e_KernelConstantTexture<AB>*)Data)->f; \
		break; \
	case e_KernelDotsTexture_TYPE : \
		r ((e_KernelDotsTexture<AB, STD_TEX_SIZE, STD_TEX_SIZE>*)Data)->f; \
		break; \
	case e_KernelFbmTexture_TYPE : \
		r ((e_KernelFbmTexture<AB>*)Data)->f; \
		break; \
	case e_KernelImageTexture_TYPE : \
		r ((e_KernelImageTexture<AB>*)Data)->f; \
		break; \
	case e_KernelMixTexture_TYPE : \
		r ((e_KernelMixTexture<AB, STD_TEX_SIZE, STD_TEX_SIZE, STD_TEX_SIZE>*)Data)->f; \
		break; \
	case e_KernelWindyTexture_TYPE : \
		r ((e_KernelWindyTexture<AB>*)Data)->f; \
		break; \
	case e_KernelWrinkledTexture_TYPE : \
		r ((e_KernelWrinkledTexture<AB>*)Data)->f; \
		break; \
	}

template <typename T, int SIZE> struct e_KernelTextureGE
{
public:
	unsigned char Data[SIZE];
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

template <int SIZE> struct e_KernelTextureGE<float3, SIZE>
{
public:
	unsigned char Data[SIZE];
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
	CUDA_FUNC_IN float3 Evaluate(const MapParameters & dg) const
	{
		CALL_FUNC(float3, return, Evaluate(dg))
	}
	template<typename L> void LoadTextures(L callback)
	{
		CALL_FUNC(float3, , LoadTextures(callback))
	}
};

#undef CALL_TYPE
#undef CALL_FUNC
#undef CALL_FUNCO

template <typename T> struct e_KernelTexture : public e_KernelTextureGE<T, STD_TEX_SIZE>
{

};

template<typename T> static inline e_KernelTexture<T> CreateTexture(const T& val)
{
	e_KernelTexture<T> r;
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