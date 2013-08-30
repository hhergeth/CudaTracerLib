#pragma once

#include <MathTypes.h>
#include "Engine\e_KernelTexture.h"
#include "Engine/e_Samples.h"
#include "Engine\e_PhaseFunction.h"

class MicrofacetDistribution
{
	enum EType {
		/// Beckmann distribution derived from Gaussian random surfaces
		EBeckmann         = 0,
		/// Long-tailed distribution proposed by Walter et al.
		EGGX              = 1,
		/// Classical Phong distribution
		EPhong            = 2,
		/// Anisotropic distribution by Ashikhmin and Shirley
		EAshikhminShirley = 3
	};
	EType m_type;
public:
	CUDA_FUNC_IN float transformRoughness(float value) const {
		value = MAX(value, (float) 1e-3f);
		if (m_type == EPhong || m_type == EAshikhminShirley)
			value = MAX(2 / (value * value) - 2, (float) 0.1f);
		return value;
	}
	CUDA_FUNC_IN float eval(const float3 &m, float alpha) const {
		return eval(m, alpha, alpha);
	}
	CUDA_DEVICE CUDA_HOST float eval(const float3 &m, float alphaU, float alphaV) const;
	CUDA_FUNC_IN float pdf(const float3 &m, float alpha) const {
		return pdf(m, alpha, alpha);
	}
	CUDA_DEVICE CUDA_HOST float pdf(const float3 &m, float alphaU, float alphaV) const;
	CUDA_FUNC_IN void sampleFirstQuadrant(float alphaU, float alphaV, float u1, float u2,
			float &phi, float &cosTheta) const {
		if (alphaU == alphaV)
			phi = PI * u1 * 0.5f;
		else
			phi = std::atan(
				sqrtf((alphaU + 1.0f) / (alphaV + 1.0f)) *
				std::tan(PI * u1 * 0.5f));
		const float cosPhi = std::cos(phi), sinPhi = std::sin(phi);
		cosTheta = std::pow(u2, 1.0f /
			(alphaU * cosPhi * cosPhi + alphaV * sinPhi * sinPhi + 1.0f));
	}
	CUDA_FUNC_IN float3 sample(const float2 &sample, float alpha) const {
		return MicrofacetDistribution::sample(sample, alpha, alpha);
	}
	CUDA_DEVICE CUDA_HOST float3 sample(const float2 &sample, float alphaU, float alphaV) const;
	CUDA_DEVICE CUDA_HOST float3 sample(const float2 &sample, float alphaU, float alphaV, float &pdf) const;
	CUDA_DEVICE CUDA_HOST float smithG1(const float3 &v, const float3 &m, float alpha) const;
	CUDA_FUNC_IN float G(const float3 &wi, const float3 &wo, const float3 &m, float alpha) const {
		return G(wi, wo, m, alpha, alpha);
	}
	CUDA_DEVICE CUDA_HOST float G(const float3 &wi, const float3 &wo, const float3 &m, float alphaU, float alphaV) const;
};

struct BSDF
{
	unsigned int m_combinedType;
	CUDA_FUNC_IN unsigned int getType()
	{
		return m_combinedType;
	}
	CUDA_FUNC_IN bool hasComponent(unsigned int type) const {
		return (type & m_combinedType) != 0;
	}
	BSDF(unsigned int type)
		: m_combinedType(type)
	{
	}
};

#include "e_BSDF_Simple.h"
#include "e_BSDF_Complex.h"

struct BSDFALL
{		/*CALL_TYPE(coating, f, r) \
		CALL_TYPE(roughcoating, f, r) \
		CALL_TYPE(mixturebsdf, f, r) \
		CALL_TYPE(blend, f, r) \*/
#define CALL_TYPE(t,f,r) \
	case t##_TYPE : \
		r ((t*)Data)->f; \
		break;
#define CALL_FUNC(r,f) \
	switch (m_uType) \
	{ \
		CALL_TYPE(diffuse, f, r) \
		CALL_TYPE(roughdiffuse, f, r) \
		CALL_TYPE(dielectric, f, r) \
		CALL_TYPE(thindielectric, f, r) \
		CALL_TYPE(roughdielectric, f, r) \
		CALL_TYPE(conductor, f, r) \
		CALL_TYPE(roughconductor, f, r) \
		CALL_TYPE(plastic, f, r) \
		CALL_TYPE(phong, f, r) \
		CALL_TYPE(ward, f, r) \
		CALL_TYPE(hk, f, r) \
	}
private:
	CUDA_ALIGN(16) unsigned char Data[255];
	unsigned int m_uType;
public:
	CUDA_FUNC_IN float3 sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const
	{
		CALL_FUNC(return, sample(bRec, pdf, sample));
		return make_float3(0);
	}
	CUDA_FUNC_IN float3 f(const BSDFSamplingRecord &bRec, EMeasure measure) const
	{
		CALL_FUNC(return, f(bRec, measure));
		return make_float3(0);
	}
	CUDA_FUNC_IN float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const
	{
		CALL_FUNC(return, pdf(bRec, measure));
		return 0.0f;
	}
	template<typename T> void LoadTextures(T callback) const
	{
		CALL_FUNC(, LoadTextures(callback));
	}
	CUDA_FUNC_IN unsigned int getType()
	{
		CALL_FUNC(return, getType());
		return 0;
	}
	CUDA_FUNC_IN bool hasComponent(unsigned int type) const {
		CALL_FUNC(return, hasComponent(type));
		return false;
	}
	template<typename T> T* As()
	{
		return (T*)Data;
	}
	template<typename T> void SetData(const T& val)
	{
		memcpy(Data, &val, sizeof(T));
		m_uType = T::TYPE();
	}
#undef CALL_FUNC
#undef CALL_TYPE
};