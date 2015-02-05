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
		: BSDF(EDeltaReflection, 0, &m_sigmaA, &m_specularReflectance)
	{
	}
	coating(BSDFFirst& nested, float eta, float thickness, const e_KernelTexture& sig)
		: BSDF(EBSDFType(EDeltaReflection | nested.getType()), new std::vector<e_KernelTexture*>(nested.As()->getTextureList()), &m_sigmaA, &m_specularReflectance), m_nested(nested), m_eta(eta), m_invEta(1.0f / eta), m_thickness(thickness), m_sigmaA(sig)
	{
		m_specularReflectance = CreateTexture(0, Spectrum(1.0f));
		float avgAbsorption = (m_sigmaA.Average()*(-2*m_thickness)).exp().average();
		m_specularSamplingWeight = 1.0f / (avgAbsorption + 1.0f);
	}
	coating(BSDFFirst& nested, float eta, float thickness, const e_KernelTexture& sig, const e_KernelTexture& specular)
		: BSDF(EBSDFType(EDeltaReflection | nested.getType()), new std::vector<e_KernelTexture*>(nested.As()->getTextureList()), &m_sigmaA, &m_specularReflectance), m_nested(nested), m_eta(eta), m_invEta(1.0f / eta), m_thickness(thickness), m_sigmaA(sig)
	{
		m_specularReflectance = specular;
		float avgAbsorption = (m_sigmaA.Average()*(-2*m_thickness)).exp().average();
		m_specularSamplingWeight = 1.0f / (avgAbsorption + 1.0f);
	}
	virtual void Update()
	{
		this->m_combinedType = EDeltaReflection | m_nested.getType();
		m_invEta = 1.0f / m_eta;
		float avgAbsorption = (m_sigmaA.Average()*(-2*m_thickness)).exp().average();
		m_specularSamplingWeight = 1.0f / (avgAbsorption + 1.0f);
	}
	CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const Vec2f &sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	template<typename T> static coating Create(const T& val, float eta, float thickness, e_KernelTexture& sig)
	{
		BSDFFirst nested;
		nested.SetData(val);
		return coating(nested, eta, thickness, sig);
	}
	TYPE_FUNC(coating)
private:
	CUDA_FUNC_IN Vec3f reflect(const Vec3f &wi) const {
		return Vec3f(-wi.x, -wi.y, wi.z);
	}
	/// Refract into the material, preserve sign of direction
	CUDA_FUNC_IN Vec3f refractIn(const Vec3f &wi, float &R) const {
		float cosThetaT;
		R = MonteCarlo::fresnelDielectricExt(math::abs(Frame::cosTheta(wi)), cosThetaT, m_eta);
		return Vec3f(m_invEta*wi.x, m_invEta*wi.y, -math::sign(Frame::cosTheta(wi)) * cosThetaT);
	}
	/// Refract out of the material, preserve sign of direction
	CUDA_FUNC_IN Vec3f refractOut(const Vec3f &wi, float &R) const {
		float cosThetaT;
		R = MonteCarlo::fresnelDielectricExt(math::abs(Frame::cosTheta(wi)), cosThetaT, m_invEta);
		return Vec3f(m_eta*wi.x, m_eta*wi.y, -math::sign(Frame::cosTheta(wi)) * cosThetaT);
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
		: BSDF(EGlossyReflection, 0, &m_sigmaA, &m_specularReflectance, &m_alpha)
	{
	}
	roughcoating(BSDFFirst& nested, MicrofacetDistribution::EType type, float eta, float thickness, e_KernelTexture& sig, e_KernelTexture& alpha, e_KernelTexture& specular)
		: BSDF(EBSDFType(EGlossyReflection | nested.getType()), new std::vector<e_KernelTexture*>(nested.As()->getTextureList()), &m_sigmaA, &m_specularReflectance, &m_alpha), m_nested(nested), m_eta(eta), m_invEta(1.0f / eta), m_thickness(thickness), m_sigmaA(sig), m_alpha(alpha)
	{
		m_distribution.m_type = type;
		m_specularReflectance = specular;
		float avgAbsorption = (m_sigmaA.Average()*(-2*m_thickness)).exp().average();
		m_specularSamplingWeight = 1.0f / (avgAbsorption + 1.0f);
	}
	virtual void Update()
	{
		this->m_combinedType = EGlossyReflection | m_nested.getType();
		m_invEta = 1.0f / m_eta;
		float avgAbsorption = (m_sigmaA.Average()*(-2*m_thickness)).exp().average();
		m_specularSamplingWeight = 1.0f / (avgAbsorption + 1.0f);
	}
	CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const Vec2f &sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	template<typename T> static roughcoating Create(const T& val, MicrofacetDistribution::EType type, float eta, float thickness, e_KernelTexture& sig, e_KernelTexture& alpha)
	{
		BSDFFirst nested;
		nested.SetData(val);
		return roughcoating(nested, type, eta, thickness, sig, alpha, CreateTexture(Spectrum(1.0f)));
	}
	TYPE_FUNC(roughcoating)
private:
	/// Helper function: reflect \c wi with respect to a given surface normal
	CUDA_FUNC_IN Vec3f reflect(const Vec3f &wi, const Vec3f &m) const {
		return 2 * dot(wi, m) * m - wi;
	}
	/// Refraction in local coordinates
	CUDA_FUNC_IN Vec3f refractTo(EDestination dest, const Vec3f &wi) const {
		float cosThetaI = Frame::cosTheta(wi);
		float invEta = (dest == EInterior) ? m_invEta : m_eta;

		bool entering = cosThetaI > 0.0f;

		/* Using Snell's law, calculate the squared sine of the
		   angle between the normal and the transmitted ray */
		float sinThetaTSqr = invEta*invEta * Frame::sinTheta2(wi);

		if (sinThetaTSqr >= 1.0f) {
			/* Total internal reflection */
			return Vec3f(0.0f);
		} else {
			float cosThetaT = math::sqrt(1.0f - sinThetaTSqr);

			/* Retain the directionality of the vector */
			return Vec3f(invEta*wi.x, invEta*wi.y,
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
	CUDA_FUNC_IN float pdf(Vec3f& a, Vec3f& b);
};
*/
#define blend_TYPE 16
struct blend : public BSDF
{
	static std::vector<e_KernelTexture*>* join(const BSDFFirst& nested1, const BSDFFirst& nested2)
	{
		std::vector<e_KernelTexture*> q, a = nested1.As()->getTextureList(), b = nested2.As()->getTextureList();
		q.insert(q.end(), a.begin(), a.end());
		q.insert(q.end(), b.begin(), b.end());
		return new std::vector<e_KernelTexture*>(q);
	}

	BSDFFirst bsdfs[2];
	e_KernelTexture weight;
public:
	blend()
		: BSDF(EBSDFType(0), 0, &weight)
	{
	}
	blend(const BSDFFirst& nested1, const BSDFFirst& nested2, const e_KernelTexture& _weight)
		: BSDF(EBSDFType(nested1.getType() | nested2.getType()), join(nested1, nested2), &weight), weight(_weight)
	{
		bsdfs[0] = nested1;
		bsdfs[1] = nested2;
	}
	CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const Vec2f &sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	template<typename U, typename V> static blend Create(const U& a, const V& b, const e_KernelTexture& weight)
	{
		BSDFFirst n1, n2;
		n1.SetData(a);
		n2.SetData(b);
		return blend(n1, n2, weight);
	}
	TYPE_FUNC(blend)
};