#pragma once

#include "e_Texture.h"
#include "..\Math\vector.h"
#include "e_Brdf.h"
#include "e_Sampler.h"

#define e_KernelMaterial_Glass_TYPE 1
struct e_KernelMaterial_Glass
{
	e_KernelMaterial_Glass(e_Sampler<float3> kr, e_Sampler<float3> kt, e_Sampler<float> index)
		: host_Kr(kr), host_Kt(kt), host_Index(index)
	{
	}

	CUDA_FUNC_IN void GetBSDF(const float2& uv, e_KernelBSDF* r) const
	{
		float ior = host_Index.Sample(uv);
		float3 R =  host_Kr.Sample(uv);
		float3 T =  host_Kt.Sample(uv);
		if(!ISBLACK(R))
			r->Add(e_KernelBrdf_SpecularReflection(R, e_KernelFresnel(1.0f, ior)));
		if(!ISBLACK(T))
			r->Add(e_KernelBrdf_SpecularTransmission(T, 1.0f, ior));
	}

	CUDA_FUNC_IN bool GetBSSRDF(const float2& uv, e_KernelBSSRDF* res) const
	{
		return false;
	}

	template<typename L> void LoadTextures(L callback)
	{
		host_Kr.LoadTextures(callback);
		host_Kt.LoadTextures(callback);
		host_Index.LoadTextures(callback);
	}

	TYPE_FUNC(e_KernelMaterial_Glass)
public:
	e_Sampler<float3> host_Kr;
	e_Sampler<float3> host_Kt;
	e_Sampler<float> host_Index;
};

#define e_KernelMaterial_Matte_TYPE 2
struct e_KernelMaterial_Matte
{
	e_KernelMaterial_Matte(e_Sampler<float3> kd, e_Sampler<float> sigma)
		: host_Kd(kd), host_Sigma(sigma)
	{
	}

	CUDA_FUNC_IN void GetBSDF(const float2& uv, e_KernelBSDF* r) const
	{
		float sigma = host_Sigma.Sample(uv);
		float3 R = host_Kd.Sample(uv);
		if(sigma == 0)
			r->Add(e_KernelBrdf_Lambertain(R));
		else r->Add(e_KernelBrdf_OrenNayar(R, sigma));
	}

	CUDA_FUNC_IN bool GetBSSRDF(const float2& uv, e_KernelBSSRDF* res) const
	{
		return false;
	}

	template<typename L> void LoadTextures(L callback)
	{
		host_Kd.LoadTextures(callback);
		host_Sigma.LoadTextures(callback);
	}

	TYPE_FUNC(e_KernelMaterial_Matte)
public:
	e_Sampler<float3> host_Kd;
	e_Sampler<float> host_Sigma;
};

#define e_KernelMaterial_Mirror_TYPE 3
struct e_KernelMaterial_Mirror
{
	e_KernelMaterial_Mirror(e_Sampler<float3> kr)
		: host_Kr(kr)
	{
	}

	CUDA_FUNC_IN void GetBSDF(const float2& uv, e_KernelBSDF* r) const
	{
		float3 R = host_Kr.Sample(uv);
		if(!ISBLACK(R))
			r->Add(e_KernelBrdf_SpecularReflection(R, e_KernelFresnel()));
	}

	CUDA_FUNC_IN bool GetBSSRDF(const float2& uv, e_KernelBSSRDF* res) const
	{
		return false;
	}

	template<typename L> void LoadTextures(L callback)
	{
		host_Kr.LoadTextures(callback);
	}

	TYPE_FUNC(e_KernelMaterial_Mirror)
public:
	e_Sampler<float3> host_Kr;
};

#define e_KernelMaterial_Metal_TYPE 4
struct e_KernelMaterial_Metal
{
	e_KernelMaterial_Metal(e_Sampler<float3> eta, e_Sampler<float3> k, e_Sampler<float> roughness)
		: host_Eta(eta), host_K(k), host_Roughness(roughness)
	{
	}

	CUDA_FUNC_IN void GetBSDF(const float2& uv, e_KernelBSDF* r) const
	{
		float rough = host_Roughness.Sample(uv);
		float3 e = host_Eta.Sample(uv);
		float3 k = host_K.Sample(uv);
		r->Add(e_KernelBrdf_Blinn(make_float3(1), e_KernelFresnel(e, k), e_KernelBrdf_BlinnDistribution(1.0f / rough)));
	}

	CUDA_FUNC_IN bool GetBSSRDF(const float2& uv, e_KernelBSSRDF* res) const
	{
		return false;
	}

	template<typename L> void LoadTextures(L callback)
	{
		host_Eta.LoadTextures(callback);
		host_K.LoadTextures(callback);
		host_Roughness.LoadTextures(callback);
	}

	TYPE_FUNC(e_KernelMaterial_Metal)
public:
	e_Sampler<float3> host_Eta;
	e_Sampler<float3> host_K;
	e_Sampler<float> host_Roughness;
};

#define e_KernelMaterial_ShinyMetal_TYPE 5
struct e_KernelMaterial_ShinyMetal
{
	e_KernelMaterial_ShinyMetal(e_Sampler<float3> ks, e_Sampler<float3> kr, e_Sampler<float> roughness)
		: host_Ks(ks), host_Kr(kr), host_Roughness(roughness)
	{
	}

	CUDA_FUNC_IN void GetBSDF(const float2& uv, e_KernelBSDF* r) const
	{
		float rough = host_Roughness.Sample(uv);
		float3 spec = host_Ks.Sample(uv);
		float3 R = host_Kr.Sample(uv);
		float3 k = make_float3(0);
		r->Add(e_KernelBrdf_Blinn(make_float3(1), e_KernelFresnel(FresnelApproxEta(spec), k), e_KernelBrdf_BlinnDistribution(1.0f / rough)));
		r->Add(e_KernelBrdf_SpecularReflection(make_float3(1), e_KernelFresnel(FresnelApproxEta(R), k)));
	}

	CUDA_FUNC_IN bool GetBSSRDF(const float2& uv, e_KernelBSSRDF* res) const
	{
		return false;
	}

	template<typename L> void LoadTextures(L callback)
	{
		host_Ks.LoadTextures(callback);
		host_Kr.LoadTextures(callback);
		host_Roughness.LoadTextures(callback);
	}

	TYPE_FUNC(e_KernelMaterial_ShinyMetal)
public:
	e_Sampler<float3> host_Ks;
	e_Sampler<float3> host_Kr;
	e_Sampler<float> host_Roughness;
private:
	CUDA_FUNC_IN float3 FresnelApproxEta(const float3 &Fr) const
	{
		float3 reflectance = clamp(Fr, 0, 0.9999f);
		return (make_float3(1.0f) + fsqrtf(reflectance)) / (make_float3(1.0f) - fsqrtf(reflectance));
	}

	CUDA_FUNC_IN float3 FresnelApproxK(const float3 &Fr) const
	{
		float3 reflectance = clamp(Fr, 0, 0.9999f);
		return 2.f * fsqrtf(reflectance / (make_float3(1.0f) - reflectance));
	}
};

#define e_KernelMaterial_Plastic_TYPE 6
struct e_KernelMaterial_Plastic
{
	e_KernelMaterial_Plastic(e_Sampler<float3> kd, e_Sampler<float3> ks, e_Sampler<float> roughness)
		: host_Kd(kd), host_Ks(ks), host_Roughness(roughness)
	{
	}

	CUDA_FUNC_IN void GetBSDF(const float2& uv, e_KernelBSDF* r) const
	{
		float rough = host_Roughness.Sample(uv);
		float3 kd = host_Kd.Sample(uv);
		float3 ks = host_Ks.Sample(uv);
		r->Add(e_KernelBrdf_Blinn(ks, e_KernelFresnel(1.5f, 1.f), e_KernelBrdf_BlinnDistribution(1.0f / rough)));
		r->Add(e_KernelBrdf_Lambertain(kd));
	}

	CUDA_FUNC_IN bool GetBSSRDF(const float2& uv, e_KernelBSSRDF* res) const
	{
		return false;
	}

	template<typename L> void LoadTextures(L callback)
	{
		host_Kd.LoadTextures(callback);
		host_Ks.LoadTextures(callback);
		host_Roughness.LoadTextures(callback);
	}

	TYPE_FUNC(e_KernelMaterial_Plastic)
public:
	e_Sampler<float3> host_Kd;
	e_Sampler<float3> host_Ks;
	e_Sampler<float> host_Roughness;
};

#define e_KernelMaterial_Substrate_TYPE 7
struct e_KernelMaterial_Substrate
{
	e_KernelMaterial_Substrate(e_Sampler<float3> kd, e_Sampler<float3> ks, e_Sampler<float> nu, e_Sampler<float> nv)
		: host_Kd(kd), host_Ks(ks), host_Nu(nu), host_Nv(nv)
	{
	}

	CUDA_FUNC_IN void GetBSDF(const float2& uv, e_KernelBSDF* r) const
	{
		float nu = host_Nu.Sample(uv);
		float nv = host_Nv.Sample(uv);
		float3 kd = host_Kd.Sample(uv);
		float3 ks = host_Ks.Sample(uv);
		r->Add(e_KernelBrdf_FresnelBlend(kd, ks, e_KernelBrdf_MicrofacetDistribution(e_KernelBrdf_AnisotropicDistribution(1.0f/nu, 1.0f/nv))));
	}

	CUDA_FUNC_IN bool GetBSSRDF(const float2& uv, e_KernelBSSRDF* res) const
	{
		return false;
	}

	template<typename L> void LoadTextures(L callback)
	{
		host_Kd.LoadTextures(callback);
		host_Ks.LoadTextures(callback);
		host_Nu.LoadTextures(callback);
		host_Nv.LoadTextures(callback);
	}

	TYPE_FUNC(e_KernelMaterial_Substrate)
public:
	e_Sampler<float3> host_Kd;
	e_Sampler<float3> host_Ks;
	e_Sampler<float> host_Nu;
	e_Sampler<float> host_Nv;
};

#define e_KernelMaterial_KdSubsurface_TYPE 8
struct e_KernelMaterial_KdSubsurface
{
	e_KernelMaterial_KdSubsurface(e_Sampler<float3> kd, e_Sampler<float3> kr, e_Sampler<float> meanfreepath, e_Sampler<float> eta)
		: host_Kd(kd), host_Kr(kr), host_MeanFreePath(meanfreepath), host_Eta(eta)
	{
	}

	CUDA_FUNC_IN void GetBSDF(const float2& uv, e_KernelBSDF* r) const
	{
		float e = host_Eta.Sample(uv);
		float3 kr = host_Kr.Sample(uv);
		if(!ISBLACK(kr))
			r->Add(e_KernelBrdf_SpecularReflection(kr, e_KernelFresnel(1.0f, e)));
	}

	CUDA_FUNC_IN bool GetBSSRDF(const float2& uv, e_KernelBSSRDF* res) const
	{
		float e = host_Eta.Sample(uv);
		float mfp = host_MeanFreePath.Sample(uv);
		float3 kd = host_Kd.Sample(uv);
		float3 sigma_a, sigma_prime_s;
		SubsurfaceFromDiffuse(kd, mfp, e, &sigma_a, &sigma_prime_s);
		*res = e_KernelBSSRDF(e, sigma_a, sigma_prime_s);
		return true;
	}

	template<typename L> void LoadTextures(L callback)
	{
		host_Kd.LoadTextures(callback);
		host_Kr.LoadTextures(callback);
		host_MeanFreePath.LoadTextures(callback);
		host_Eta.LoadTextures(callback);
	}

	TYPE_FUNC(e_KernelMaterial_KdSubsurface)
public:
	e_Sampler<float3> host_Kd;
	e_Sampler<float3> host_Kr;
	e_Sampler<float> host_MeanFreePath;
	e_Sampler<float> host_Eta;
};

#define e_KernelMaterial_Subsurface_TYPE 9
struct e_KernelMaterial_Subsurface
{
	e_KernelMaterial_Subsurface(e_Sampler<float3> kr, e_Sampler<float3> sigma_a, e_Sampler<float3> Sigma_prime_s, e_Sampler<float> eta, float _scale = 1.0f)
		: host_Kr(kr), host_Sigma_a(sigma_a), host_Sigma_prime_s(Sigma_prime_s), host_Eta(eta), scale(_scale)
	{
	}

	CUDA_FUNC_IN void GetBSDF(const float2& uv, e_KernelBSDF* r) const
	{
		float e = host_Eta.Sample(uv);
		float3 kr = host_Kr.Sample(uv);
		if(!ISBLACK(kr))
			r->Add(e_KernelBrdf_SpecularReflection(kr, e_KernelFresnel(1.0f, e)));
			//r->Add(CREATE_e_KernelBXDFL(e_KernelBrdf_Lambertain, e_KernelBrdf_Lambertain(kr)));
	}

	CUDA_FUNC_IN bool GetBSSRDF(const float2& uv, e_KernelBSSRDF* res) const
	{
		float e = host_Eta.Sample(uv);
		*res = e_KernelBSSRDF(e, host_Sigma_a.Sample(uv) * scale, host_Sigma_prime_s.Sample(uv) * scale);
		return true;
	}

	template<typename L> void LoadTextures(L callback)
	{
		host_Kr.LoadTextures(callback);
		host_Sigma_a.LoadTextures(callback);
		host_Sigma_prime_s.LoadTextures(callback);
		host_Eta.LoadTextures(callback);
	}

	TYPE_FUNC(e_KernelMaterial_Subsurface)
public:
	e_Sampler<float3> host_Kr;
	e_Sampler<float3> host_Sigma_a;
	e_Sampler<float3> host_Sigma_prime_s;
	e_Sampler<float> host_Eta;
	float scale;
};

struct e_KernelMaterial
{
public:
	unsigned char Data[2048];
	unsigned int type;
#define CALL_TYPE(t,f,r) \
	case t##_TYPE : \
		r ((t*)Data)->f; \
		break;
#define CALL_FUNC(r,f) \
	switch (type) \
	{ \
		CALL_TYPE(e_KernelMaterial_Glass, f, r) \
		CALL_TYPE(e_KernelMaterial_Matte, f, r) \
		CALL_TYPE(e_KernelMaterial_Mirror, f, r) \
		CALL_TYPE(e_KernelMaterial_Metal, f, r) \
		CALL_TYPE(e_KernelMaterial_ShinyMetal, f, r) \
		CALL_TYPE(e_KernelMaterial_Plastic, f, r) \
		CALL_TYPE(e_KernelMaterial_Substrate, f, r) \
		CALL_TYPE(e_KernelMaterial_KdSubsurface, f, r) \
		CALL_TYPE(e_KernelMaterial_Subsurface, f, r) \
	}
public:
	e_String Name;
	unsigned int NodeLightIndex;
	e_Sampler<float3> NormalMap;
	e_Sampler<float3> HeightMap;
	float HeightScale;
public:
	e_KernelMaterial()
		: NormalMap(make_float3(0)), HeightMap(make_float3(0))
	{
		HeightScale = 1.0f;
		memset(Name, 0, sizeof(Name));
	}
	e_KernelMaterial(const char* name)
		: NormalMap(make_float3(0)), HeightMap(make_float3(0))
	{
		memset(Name, 0, sizeof(Name));
		HeightScale = 1.0f;
		memcpy(Name, name, strlen(name));
		NodeLightIndex = -1;
	}
	template<typename T> void SetData(T& val)
	{
		*(T*)Data = val;
		type = T::TYPE();
	}

	CUDA_FUNC_IN void GetBSDF(const float2& uv, e_KernelBSDF* res) const
	{
#ifndef FAST_BRDF
		CALL_FUNC(, GetBSDF(uv, res))
#endif
	}

	CUDA_FUNC_IN bool GetBSSRDF(const float2& uv, e_KernelBSSRDF* res) const
	{
		CALL_FUNC(return, GetBSSRDF(uv, res))
	}

	template<typename L> void LoadTextures(L callback)
	{
		NormalMap.LoadTextures(callback);
		HeightMap.LoadTextures(callback);
		CALL_FUNC(return, LoadTextures<L>(callback))
	}

	template<typename T> T* As()
	{
		return (T*)Data;
	}

	CUDA_FUNC_IN bool SampleNormalMap(const float2& uv, float3* normal) const
	{
		if(NormalMap.HasTexture())
		{
			*normal = NormalMap.Sample(uv) * 2.0f - make_float3(1);
			return true;
		}
		else if(HeightMap.HasTexture())
		{
			float m[16];
			HeightMap.m_pTex->Gather<4>(uv, m);
			//float2 d = make_float2(m[2] + 2*m[5] + m[8] -m[0] - 2*m[3] - m[6], m[6] + 2*m[7] + m[8] -m[0] - 2*m[1] - m[2]);
			/*float2 lu = dxdy(m, 0, 1, 2, 4, 5, 6, 8, 9, 10),
				   ru = dxdy(m, 1, 2, 3, 5, 6, 7, 9, 10, 11),
				   ld = dxdy(m, 4, 5, 6, 8, 9, 10, 12, 13, 14),
				   rd = dxdy(m, 5, 6, 7, 9, 10, 11, 13, 14, 15);
			float2 d = bilerp(make_float2(frac(uv.x), frac(uv.y)), lu, ru, ld, rd);
			*normal = normalize(make_float3(d, 1.0f));*/
			*normal = nor(m, 4, 1, 5, 6, 9); 
			return true;
		}
		else return false;
	}
#undef CALL_TYPE
#undef CALL_FUNC
private:
	CUDA_FUNC_IN float2 dxdy(float* D, int tl, int t, int tr, int l, int ME, int r, int bl, int b, int br) const
	{
		return make_float2(D[tr] + 2.0f * D[r] + D[br] - D[tl] - 2.0f * D[l] - D[bl], D[bl] + 2.0f * D[b] + D[br] - D[tl] - 2.0f * D[t] - D[tr]);
	}
	CUDA_FUNC_IN float3 nor(float* D, int l, int t, int m, int r, int b) const
	{
		//return normalize(cross(make_float3(0, -1, D[t] - D[m]), make_float3(-1, 0, D[l] - D[m])) + cross(make_float3(0, 1, D[b] - D[m]), make_float3(1, 0, D[r] - D[m])));
		return normalize(make_float3(D[m]-D[l], D[m]-D[t], HeightScale));
	}
};