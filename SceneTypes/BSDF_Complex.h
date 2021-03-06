#pragma once

#include "BSDF_Simple.h"

namespace CudaTracerLib
{
struct BSDFALL;

struct coating : public BSDF//, public e_DerivedTypeHelper<13>
{
	TYPE_FUNC(13)
	BSDFFirst m_nested;
	float m_specularSamplingWeight;
	float m_eta, m_invEta;
	Texture m_sigmaA;
	Texture m_specularReflectance;
	float m_thickness;
	coating()
		: BSDF(EDeltaReflection)
	{
		BSDF::initTextureOffsets(m_sigmaA, m_specularReflectance);
	}
	coating(const BSDFFirst& nested, float eta, float thickness, const Texture& sig)
		: BSDF(EBSDFType(EDeltaReflection | nested.getType())), m_nested(nested), m_eta(eta), m_invEta(1.0f / eta), m_thickness(thickness), m_sigmaA(sig), m_specularReflectance(CreateTexture(Spectrum(1.0f)))
	{
		BSDF::initTextureOffsets2(m_nested.As()->getTextureList(), m_sigmaA, m_specularReflectance);
		float avgAbsorption = (m_sigmaA.Average()*(-2*m_thickness)).exp().avg();
		m_specularSamplingWeight = 1.0f / (avgAbsorption + 1.0f);
	}
	coating(const BSDFFirst& nested, float eta, float thickness, const Texture& sig, const Texture& specular)
		: BSDF(EBSDFType(EDeltaReflection | nested.getType())), m_nested(nested), m_eta(eta), m_invEta(1.0f / eta), m_thickness(thickness), m_sigmaA(sig), m_specularReflectance(specular)
	{
		BSDF::initTextureOffsets2(m_nested.As()->getTextureList(), m_sigmaA, m_specularReflectance);
		float avgAbsorption = (m_sigmaA.Average()*(-2*m_thickness)).exp().avg();
		m_specularSamplingWeight = 1.0f / (avgAbsorption + 1.0f);
	}
	virtual void Update()
	{
		BSDF::initTextureOffsets2(m_nested.As()->getTextureList(), m_sigmaA, m_specularReflectance);
		this->m_combinedType = EDeltaReflection | m_nested.getType();
		m_invEta = 1.0f / m_eta;
		float avgAbsorption = (m_sigmaA.Average()*(-2*m_thickness)).exp().avg();
		m_specularSamplingWeight = 1.0f / (avgAbsorption + 1.0f);
	}
	CTL_EXPORT CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const Vec2f &sample) const;
	CTL_EXPORT CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CTL_EXPORT CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	template<typename T> static coating Create(const T& val, float eta, float thickness, Texture& sig)
	{
		BSDFFirst nested;
		nested.SetData(val);
		return coating(nested, eta, thickness, sig);
	}
private:
	/// Refract into the material, preserve sign of direction
	CUDA_FUNC_IN NormalizedT<Vec3f> refractIn(const NormalizedT<Vec3f> &wi, float &R) const {
		float cosThetaT;
		R = FresnelHelper::fresnelDielectricExt(math::abs(Frame::cosTheta(wi)), cosThetaT, m_eta);
		return Vec3f(m_invEta*wi.x, m_invEta*wi.y, -math::sign(Frame::cosTheta(wi)) * cosThetaT).normalized();
	}
	/// Refract out of the material, preserve sign of direction
	CUDA_FUNC_IN NormalizedT<Vec3f> refractOut(const NormalizedT<Vec3f> &wi, float &R) const {
		float cosThetaT;
		R = FresnelHelper::fresnelDielectricExt(math::abs(Frame::cosTheta(wi)), cosThetaT, m_invEta);
		return Vec3f(m_eta*wi.x, m_eta*wi.y, -math::sign(Frame::cosTheta(wi)) * cosThetaT).normalized();
	}
};

struct roughcoating : public BSDF//, public e_DerivedTypeHelper<14>
{
	TYPE_FUNC(14)
	enum EDestination {
		EInterior = 0,
		EExterior = 1
	};

	BSDFFirst m_nested;
	Texture m_sigmaA;
	Texture m_alpha;
	Texture m_specularReflectance;
	float m_specularSamplingWeight;
	float m_eta, m_invEta;
	float m_thickness;
	bool m_sampleVisible;
	MicrofacetDistribution::EType m_type;

	roughcoating()
		: BSDF(EGlossyReflection), m_type(MicrofacetDistribution::EBeckmann)
	{
		initTextureOffsets(m_sigmaA, m_specularReflectance, m_alpha);
		Update();
	}
	roughcoating(MicrofacetDistribution::EType type, const BSDFFirst& nested, float eta, float thickness, const Texture& sig, const Texture& alpha, const Texture& specular)
		: BSDF(EBSDFType(EGlossyReflection | nested.getType())), m_nested(nested), m_eta(eta), m_invEta(1.0f / eta), m_thickness(thickness), m_sigmaA(sig), m_alpha(alpha), m_specularReflectance(specular), m_type(type)
	{
		initTextureOffsets(m_sigmaA, m_specularReflectance, m_alpha);
		Update();
	}
	virtual void Update()
	{
		initTextureOffsets2(m_nested.As()->getTextureList(), m_sigmaA, m_specularReflectance, m_alpha);
		this->m_combinedType = EGlossyReflection | m_nested.getType();
		m_invEta = 1.0f / m_eta;
		float avgAbsorption = (m_sigmaA.Average()*(-2*m_thickness)).exp().avg();
		m_specularSamplingWeight = 1.0f / (avgAbsorption + 1.0f);
		m_sampleVisible = MicrofacetDistribution::getSampleVisible(m_type, true);
	}
	CTL_EXPORT CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const Vec2f &sample) const;
	CTL_EXPORT CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CTL_EXPORT CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	template<typename T> static roughcoating Create(const T& val, MicrofacetDistribution::EType type, float eta, float thickness, Texture& sig, Texture& alpha)
	{
		BSDFFirst nested;
		nested.SetData(val);
		return roughcoating(type, nested, eta, thickness, sig, alpha, CreateTexture(Spectrum(1.0f)));
	}
private:
	/// Refraction in local coordinates
	CUDA_FUNC_IN NormalizedT<Vec3f> refractTo(EDestination dest, const NormalizedT<Vec3f> &wi) const {
		float cosThetaI = Frame::cosTheta(wi);
		float invEta = (dest == EInterior) ? m_invEta : m_eta;

		bool entering = cosThetaI > 0.0f;

		/* Using Snell's law, calculate the squared sine of the
		   angle between the normal and the transmitted ray */
		float sinThetaTSqr = invEta*invEta * Frame::sinTheta2(wi);

		if (sinThetaTSqr >= 1.0f) {
			/* Total internal reflection */
			return NormalizedT<Vec3f>(0.0f);
		} else {
			float cosThetaT = math::sqrt(1.0f - sinThetaTSqr);

			/* Retain the directionality of the vector */
			return Vec3f(invEta*wi.x, invEta*wi.y,
				entering ? cosThetaT : -cosThetaT).normalized();
		}
	}
};

struct blend : public BSDF//, public e_DerivedTypeHelper<16>
{
	TYPE_FUNC(15)
	static std::vector<Texture*> join(const BSDFFirst& nested1, const BSDFFirst& nested2)
	{
		std::vector<Texture*> q, a = nested1.As()->getTextureList(), b = nested2.As()->getTextureList();
		q.insert(q.end(), a.begin(), a.end());
		q.insert(q.end(), b.begin(), b.end());
		return q;
	}

	BSDFFirst bsdfs[2];
	Texture weight;
public:
	blend()
		: BSDF(EBSDFType(0))
	{
		initTextureOffsets(weight);
	}
	blend(const BSDFFirst& nested1, const BSDFFirst& nested2, const Texture& _weight)
		: BSDF(EBSDFType(nested1.getType() | nested2.getType())), weight(_weight)
	{
		bsdfs[0] = nested1;
		bsdfs[1] = nested2;
        Update();
	}
	CTL_EXPORT CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const Vec2f &sample) const;
	CTL_EXPORT CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CTL_EXPORT CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	template<typename U, typename V> static blend Create(const U& a, const V& b, const Texture& weight)
	{
		BSDFFirst n1, n2;
		n1.SetData(a);
		n2.SetData(b);
		return blend(n1, n2, weight);
	}
	virtual void Update()
	{
		initTextureOffsets2(join(bsdfs[0], bsdfs[1]), weight);
	}
};

}