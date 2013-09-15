#pragma once

#include "e_BSDF_Simple.h"

struct BSDFALL;

#define coating_TYPE 13
struct coating : public BSDF
{
	BSDFFirst m_nested;
	float m_specularSamplingWeight;
	float m_eta, m_invEta;
	e_KernelTexture m_sigmaA;
	e_KernelTexture m_specularReflectance;
	float m_thickness;
	coating()
		: BSDF(EDeltaReflection)
	{
	}
	coating(BSDFFirst& nested, float eta, float thickness, e_KernelTexture& sig)
		: BSDF(EDeltaReflection | nested.getType()), m_nested(nested), m_eta(eta), m_invEta(1.0f / eta), m_thickness(thickness), m_sigmaA(sig)
	{
		m_specularReflectance = CreateTexture(0, Spectrum(1.0f));
		float avgAbsorption = (m_sigmaA.Average()*(-2*m_thickness)).exp().average();
		m_specularSamplingWeight = 1.0f / (avgAbsorption + 1.0f);
	}
	CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	template<typename T> void LoadTextures(T callback)
	{
		m_sigmaA.LoadTextures(callback);
		m_specularReflectance.LoadTextures(callback);
		m_nested.LoadTextures(callback);
	}
	template<typename T> static coating Create(const T& val, float eta, float thickness, e_KernelTexture& sig)
	{
		BSDFFirst nested;
		nested.SetData(val);
		return coating(nested, eta, thickness, sig);
	}
	TYPE_FUNC(coating)
private:
	CUDA_FUNC_IN float3 reflect(const float3 &wi) const {
		return make_float3(-wi.x, -wi.y, wi.z);
	}
	/// Refract into the material, preserve sign of direction
	CUDA_FUNC_IN float3 refractIn(const float3 &wi, float &R) const {
		float cosThetaT;
		R = MonteCarlo::fresnelDielectricExt(abs(Frame::cosTheta(wi)), cosThetaT, m_eta);
		return make_float3(m_invEta*wi.x, m_invEta*wi.y, -signf(Frame::cosTheta(wi)) * cosThetaT);
	}
	/// Refract out of the material, preserve sign of direction
	CUDA_FUNC_IN float3 refractOut(const float3 &wi, float &R) const {
		float cosThetaT;
		R = MonteCarlo::fresnelDielectricExt(abs(Frame::cosTheta(wi)), cosThetaT, m_invEta);
		return make_float3(m_eta*wi.x, m_eta*wi.y, -signf(Frame::cosTheta(wi)) * cosThetaT);
	}
};

#define roughcoating_TYPE 14
struct roughcoating : public BSDF
{
	enum EDestination {
		EInterior = 0,
		EExterior = 1
	};

	BSDFFirst m_nested;
	MicrofacetDistribution m_distribution;
	e_KernelTexture m_sigmaA;
	e_KernelTexture m_alpha;
	e_KernelTexture m_specularReflectance;
	float m_specularSamplingWeight;
	float m_eta, m_invEta;
	float m_thickness;
	roughcoating()
		: BSDF(EGlossyReflection)
	{
	}
	roughcoating(BSDFFirst& nested, MicrofacetDistribution::EType type, float eta, float thickness, e_KernelTexture& sig, e_KernelTexture& alpha)
		: BSDF(EGlossyReflection | nested.getType()), m_nested(nested), m_eta(eta), m_invEta(1.0f / eta), m_thickness(thickness), m_sigmaA(sig), m_alpha(alpha)
	{
		m_distribution.m_type = type;
		m_specularReflectance = CreateTexture(0, Spectrum(1.0f));
		float avgAbsorption = (m_sigmaA.Average()*(-2*m_thickness)).exp().average();
		m_specularSamplingWeight = 1.0f / (avgAbsorption + 1.0f);
	}
	CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	template<typename T> void LoadTextures(T callback)
	{
		m_sigmaA.LoadTextures(callback);
		m_specularReflectance.LoadTextures(callback);
		m_nested.LoadTextures(callback);
		m_alpha.LoadTextures(callback);
	}
	template<typename T> static roughcoating Create(const T& val, MicrofacetDistribution::EType type, float eta, float thickness, e_KernelTexture& sig, e_KernelTexture& alpha)
	{
		BSDFFirst nested;
		nested.SetData(val);
		return roughcoating(nested, type, eta, thickness, sig, alpha);
	}
	TYPE_FUNC(roughcoating)
private:
	/// Helper function: reflect \c wi with respect to a given surface normal
	CUDA_FUNC_IN float3 reflect(const float3 &wi, const float3 &m) const {
		return 2 * dot(wi, m) * m - wi;
	}
	/// Refraction in local coordinates
	CUDA_FUNC_IN float3 refractTo(EDestination dest, const float3 &wi) const {
		float cosThetaI = Frame::cosTheta(wi);
		float invEta = (dest == EInterior) ? m_invEta : m_eta;

		bool entering = cosThetaI > 0.0f;

		/* Using Snell's law, calculate the squared sine of the
		   angle between the normal and the transmitted ray */
		float sinThetaTSqr = invEta*invEta * Frame::sinTheta2(wi);

		if (sinThetaTSqr >= 1.0f) {
			/* Total internal reflection */
			return make_float3(0.0f);
		} else {
			float cosThetaT = sqrtf(1.0f - sinThetaTSqr);

			/* Retain the directionality of the vector */
			return make_float3(invEta*wi.x, invEta*wi.y,
				entering ? cosThetaT : -cosThetaT);
		}
	}
};
/*
struct mixturebsdf : public BSDF
{
private:
	BSDFALL* bsdfs[10];
	float weights[10];
	int num;
public:
	CUDA_FUNC_IN float pdf(float3& a, float3& b);
};

struct blend : public BSDF
{
private:
	BSDFALL* bsdfs[2];
	float weight;
public:
	CUDA_FUNC_IN float pdf(float3& a, float3& b);
};
*/