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
			r->Add(CREATE_e_KernelBXDFL(e_KernelBrdf_SpecularReflection, e_KernelBrdf_SpecularReflection(R, e_KernelFresnel(1.0f, ior))));
		if(!ISBLACK(T))
			r->Add(CREATE_e_KernelBXDFL(e_KernelBrdf_SpecularTransmission, e_KernelBrdf_SpecularTransmission(T, 1.0f, ior)));
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
public:
	e_Sampler<float3> host_Kr;
	e_Sampler<float3> host_Kt;
	e_Sampler<float> host_Index;
public:
	static const unsigned int TYPE;
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
			r->Add(CREATE_e_KernelBXDFL(e_KernelBrdf_Lambertain, e_KernelBrdf_Lambertain(R)));
		else r->Add(CREATE_e_KernelBXDFL(e_KernelBrdf_OrenNayar, e_KernelBrdf_OrenNayar(R, sigma)));
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
public:
	e_Sampler<float3> host_Kd;
	e_Sampler<float> host_Sigma;
public:
	static const unsigned int TYPE;
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
			r->Add(CREATE_e_KernelBXDFL(e_KernelBrdf_SpecularReflection, e_KernelBrdf_SpecularReflection(R, e_KernelFresnel())));
	}

	CUDA_FUNC_IN bool GetBSSRDF(const float2& uv, e_KernelBSSRDF* res) const
	{
		return false;
	}

	template<typename L> void LoadTextures(L callback)
	{
		host_Kr.LoadTextures(callback);
	}
public:
	e_Sampler<float3> host_Kr;
public:
	static const unsigned int TYPE;
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
		r->Add(CREATE_e_KernelBXDFL(e_KernelBrdf_Blinn, e_KernelBrdf_Blinn(make_float3(1), e_KernelFresnel(e, k), e_KernelBrdf_BlinnDistribution(1.0f / rough))));
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
public:
	e_Sampler<float3> host_Eta;
	e_Sampler<float3> host_K;
	e_Sampler<float> host_Roughness;
public:
	static const unsigned int TYPE;
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
		r->Add(CREATE_e_KernelBXDFL(e_KernelBrdf_Blinn, e_KernelBrdf_Blinn(make_float3(1), e_KernelFresnel(FresnelApproxEta(spec), k), e_KernelBrdf_BlinnDistribution(1.0f / rough))));
		r->Add(CREATE_e_KernelBXDFL(e_KernelBrdf_SpecularReflection, e_KernelBrdf_SpecularReflection(make_float3(1), e_KernelFresnel(FresnelApproxEta(R), k))));
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
public:
	e_Sampler<float3> host_Ks;
	e_Sampler<float3> host_Kr;
	e_Sampler<float> host_Roughness;
private:
	CUDA_FUNC_IN float3 FresnelApproxEta(const float3 &Fr) const
	{
		float3 reflectance = saturate(Fr);
		return (make_float3(1.0f) + fsqrtf(reflectance)) / (make_float3(1.0f) - fsqrtf(reflectance));
	}

	CUDA_FUNC_IN float3 FresnelApproxK(const float3 &Fr) const
	{
		float3 reflectance = saturate(Fr);
		return 2.f * fsqrtf(reflectance / (make_float3(1.) - reflectance));
	}
public:
	static const unsigned int TYPE;
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
		r->Add(CREATE_e_KernelBXDFL(e_KernelBrdf_Blinn, e_KernelBrdf_Blinn(ks, e_KernelFresnel(1.5f, 1.f), e_KernelBrdf_BlinnDistribution(1.0f / rough))));
		r->Add(CREATE_e_KernelBXDFL(e_KernelBrdf_Lambertain, e_KernelBrdf_Lambertain(kd)));
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
public:
	e_Sampler<float3> host_Kd;
	e_Sampler<float3> host_Ks;
	e_Sampler<float> host_Roughness;
public:
	static const unsigned int TYPE;
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
		r->Add(CREATE_e_KernelBXDFL(e_KernelBrdf_FresnelBlend, e_KernelBrdf_FresnelBlend(kd, ks, e_KernelBrdf_MicrofacetDistribution(e_KernelBrdf_AnisotropicDistribution(1.0f/nu, 1.0f/nv)))));
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
public:
	e_Sampler<float3> host_Kd;
	e_Sampler<float3> host_Ks;
	e_Sampler<float> host_Nu;
	e_Sampler<float> host_Nv;
public:
	static const unsigned int TYPE;
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
	}
public:
	e_String Name;
	float3 Emission;
public:
	e_KernelMaterial()
	{

	}
	e_KernelMaterial(const char* name)
	{
		memcpy(Name, name, strlen(name));
		Emission = make_float3(0);
	}
	template<typename T> void SetData(T& val)
	{
		*(T*)Data = val;
		type = T::TYPE;
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
		CALL_FUNC(return, LoadTextures<L>(callback))
	}

	template<typename T> T* As()
	{
		return (T*)Data;
	}
#undef CALL_TYPE
#undef CALL_FUNC
};