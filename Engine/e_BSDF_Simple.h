#pragma once

struct diffuse : public BSDF//, public e_DerivedTypeHelper<1>
{
	TYPE_FUNC(1)
	e_Texture m_reflectance;
	diffuse()
		: BSDF(EDiffuseReflection, 0, &m_reflectance)
	{
	}
	diffuse(const e_Texture& d)
		: m_reflectance(d), BSDF(EDiffuseReflection, 0, &m_reflectance)
	{
	}
	CUDA_FUNC_IN Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const Vec2f &sample) const
	{
		if (!(bRec.typeMask & EDiffuseReflection) || Frame::cosTheta(bRec.wi) <= 0)
			return 0.0f;

		bRec.wo = Warp::squareToCosineHemisphere(sample);
		bRec.eta = 1.0f;
		bRec.sampledType = EDiffuseReflection;
		pdf = Warp::squareToCosineHemispherePdf(bRec.wo);
		return m_reflectance.Evaluate(bRec.dg);
	}
	CUDA_FUNC_IN Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const
	{
		if (!(bRec.typeMask & EDiffuseReflection) || measure != ESolidAngle
			|| Frame::cosTheta(bRec.wi) <= 0
			|| Frame::cosTheta(bRec.wo) <= 0)
			return 0.0f;

		return m_reflectance.Evaluate(bRec.dg)	* (INV_PI * Frame::cosTheta(bRec.wo));
	}
	CUDA_FUNC_IN float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const
	{
		if (!(bRec.typeMask & EDiffuseReflection) || measure != ESolidAngle
			|| Frame::cosTheta(bRec.wi) <= 0
			|| Frame::cosTheta(bRec.wo) <= 0)
			return 0.0f;

		return Warp::squareToCosineHemispherePdf(bRec.wo);
	}
};

struct roughdiffuse : public BSDF//, public e_DerivedTypeHelper<2>
{
	TYPE_FUNC(2)
	e_Texture m_reflectance;
	e_Texture m_alpha;
	bool m_useFastApprox;
	roughdiffuse()
		: BSDF(EDiffuseReflection, 0, &m_reflectance, &m_alpha)
	{
	}
	roughdiffuse(const e_Texture& r, const e_Texture& a)
		: m_reflectance(r), m_alpha(a), BSDF(EDiffuseReflection, 0, &m_reflectance, &m_alpha)
	{
	}
	CUDA_FUNC_IN Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const Vec2f &sample) const
	{
		bRec.wo = Warp::squareToCosineHemisphere(sample);
		bRec.eta = 1.0f;
		bRec.sampledType = EGlossyReflection;
		pdf = Warp::squareToCosineHemispherePdf(bRec.wo);
		return f(bRec, ESolidAngle) / pdf;
	}
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_FUNC_IN float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const
	{
		if (!(bRec.typeMask & EGlossyReflection) || measure != ESolidAngle
			|| Frame::cosTheta(bRec.wi) <= 0
			|| Frame::cosTheta(bRec.wo) <= 0)
			return 0.0f;

		return Warp::squareToCosineHemispherePdf(bRec.wo);
	}
};

struct dielectric : public BSDF//, public e_DerivedTypeHelper<3>
{
	TYPE_FUNC(3)
	float m_eta, m_invEta;
	e_Texture m_specularTransmittance;
	e_Texture m_specularReflectance;
	dielectric()
		: BSDF(EBSDFType(EDeltaReflection | EDeltaTransmission), 0, &m_specularTransmittance, &m_specularReflectance)
	{
	}
	dielectric(float e)
		: BSDF(EBSDFType(EDeltaReflection | EDeltaTransmission), 0, &m_specularTransmittance, &m_specularReflectance)
	{
		m_eta = e;
		m_invEta = 1.0f / e;
		m_specularTransmittance = CreateTexture(Spectrum(1.0f));
		m_specularReflectance = CreateTexture(Spectrum(1.0f));
	}
	dielectric(float e, const Spectrum& r, const Spectrum& t)
		: BSDF(EBSDFType(EDeltaReflection | EDeltaTransmission), 0, &m_specularTransmittance, &m_specularReflectance)
	{
		m_eta = e;
		m_invEta = 1.0f / e;
		m_specularTransmittance = CreateTexture(t);
		m_specularReflectance = CreateTexture(r);
	}
	dielectric(float e, const e_Texture& r, const e_Texture& t)
		: BSDF(EBSDFType(EDeltaReflection | EDeltaTransmission), 0, &m_specularTransmittance, &m_specularReflectance)
	{
		m_eta = e;
		m_invEta = 1.0f / e;
		m_specularTransmittance = t;
		m_specularReflectance = r;
	}
	virtual void Update()
	{
		m_invEta = 1.0f / m_eta;
	}
	/// Reflection in local coordinates
	CUDA_FUNC_IN Vec3f reflect(const Vec3f &wi) const {
		return Vec3f(-wi.x, -wi.y, wi.z);
	}
	/// Refraction in local coordinates
	CUDA_FUNC_IN Vec3f refract(const Vec3f &wi, float cosThetaT) const {
		float scale = -(cosThetaT < 0 ? m_invEta : m_eta);
		return Vec3f(scale*wi.x, scale*wi.y, cosThetaT);
	}
	CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const Vec2f &sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
};

struct thindielectric : public BSDF//, public e_DerivedTypeHelper<4>
{
	TYPE_FUNC(4)
	float m_eta;
	e_Texture m_specularTransmittance;
	e_Texture m_specularReflectance;
	thindielectric()
		: BSDF(EBSDFType(EDeltaReflection | EDeltaTransmission), 0, &m_specularTransmittance, &m_specularReflectance)
	{
	}
	thindielectric(float e)
		: BSDF(EBSDFType(EDeltaReflection | EDeltaTransmission), 0, &m_specularTransmittance, &m_specularReflectance)
	{
		m_eta = e;
		m_specularTransmittance = CreateTexture(Spectrum(1.0f));
		m_specularReflectance = CreateTexture(Spectrum(1.0f));
	}
	thindielectric(float e, const e_Texture& r, const e_Texture& t)
		: BSDF(EBSDFType(EDeltaReflection | EDeltaTransmission), 0, &m_specularTransmittance, &m_specularReflectance)
	{
		m_eta = e;
		m_specularTransmittance = t;
		m_specularReflectance = r;
	}
	CUDA_FUNC_IN Vec3f transmit(const Vec3f &wi) const {
		return -1.0f * wi;
	}
	/// Reflection in local coordinates
	CUDA_FUNC_IN Vec3f reflect(const Vec3f &wi) const {
		return Vec3f(-wi.x, -wi.y, wi.z);
	}
	CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const Vec2f &sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
};

struct roughdielectric : public BSDF//, public e_DerivedTypeHelper<5>
{
	TYPE_FUNC(5)
	MicrofacetDistribution m_distribution;
	e_Texture m_specularTransmittance;
	e_Texture  m_specularReflectance;
	e_Texture m_alphaU, m_alphaV;
	float m_eta, m_invEta;
	roughdielectric()
		: BSDF(EBSDFType(EGlossyReflection | EGlossyTransmission), 0, &m_specularTransmittance, &m_specularReflectance, &m_alphaU, &m_alphaV)
	{
	}
	roughdielectric(MicrofacetDistribution::EType type, float eta, const e_Texture& u, const e_Texture& v)
		: m_alphaU(u), m_alphaV(v), BSDF(EBSDFType(EGlossyReflection | EGlossyTransmission), 0, &m_specularTransmittance, &m_specularReflectance, &m_alphaU, &m_alphaV)
	{
		m_distribution.m_type = type;
		m_specularTransmittance = CreateTexture(Spectrum(1.0f));
		m_specularReflectance = CreateTexture(Spectrum(1.0f));
		m_eta = eta;
		m_invEta = 1.0f / eta;
	}
	roughdielectric(MicrofacetDistribution::EType type, float eta, const e_Texture& u, const e_Texture& v, const e_Texture& r, const e_Texture& t)
		: m_alphaU(u), m_alphaV(v), BSDF(EBSDFType(EGlossyReflection | EGlossyTransmission), 0, &m_specularTransmittance, &m_specularReflectance, &m_alphaU, &m_alphaV)
	{
		m_distribution.m_type = type;
		m_specularTransmittance = t;
		m_specularReflectance = r;
		m_eta = eta;
		m_invEta = 1.0f / eta;
	}
	virtual void Update()
	{
		m_invEta = 1.0f / m_eta;
	}
	CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const Vec2f &_sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
};

struct conductor : public BSDF//, public e_DerivedTypeHelper<6>
{
	TYPE_FUNC(6)
	e_Texture m_specularReflectance;
	Spectrum m_eta;
	Spectrum m_k;
	conductor()
		: BSDF(EDeltaReflection, 0, &m_specularReflectance)
	{
	}
	conductor(const Spectrum& eta, const Spectrum& k)
		: BSDF(EDeltaReflection, 0, &m_specularReflectance)
	{
		m_specularReflectance = CreateTexture(Spectrum(1.0f));
		m_eta = eta;
		m_k = k;
	}
	conductor(const Spectrum& eta, const Spectrum& k, const e_Texture& r)
		: BSDF(EDeltaReflection, 0, &m_specularReflectance)
	{
		m_specularReflectance = r;
		m_eta = eta;
		m_k = k;
	}
	/// Reflection in local coordinates
	CUDA_FUNC_IN Vec3f reflect(const Vec3f &wi) const {
		return Vec3f(-wi.x, -wi.y, wi.z);
	}
	CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const Vec2f &sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
};

struct roughconductor : public BSDF//, public e_DerivedTypeHelper<7>
{
	TYPE_FUNC(7)
	MicrofacetDistribution m_distribution;
	e_Texture m_specularReflectance;
	e_Texture m_alphaU, m_alphaV;
	Spectrum m_eta, m_k;
	roughconductor()
		: BSDF(EGlossyReflection, 0, &m_specularReflectance, &m_alphaU, &m_alphaV)
	{
	}
	roughconductor(MicrofacetDistribution::EType type, const Spectrum& eta, const Spectrum& k, const e_Texture& u, const e_Texture& v)
		: m_alphaU(u), m_alphaV(v), BSDF(EGlossyReflection, 0, &m_specularReflectance, &m_alphaU, &m_alphaV)
	{
		m_specularReflectance = CreateTexture(Spectrum(1.0f));
		m_eta = eta;
		m_k = k;
		m_distribution.m_type = type;
	}
	roughconductor(MicrofacetDistribution::EType type, const Spectrum& eta, const Spectrum& k, const e_Texture& u, const e_Texture& v, const e_Texture& r)
		: m_alphaU(u), m_alphaV(v), BSDF(EGlossyReflection, 0, &m_specularReflectance, &m_alphaU, &m_alphaV)
	{
		m_specularReflectance = r;
		m_eta = eta;
		m_k = k;
		m_distribution.m_type = type;
	}
	CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const Vec2f &sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
};

struct plastic : public BSDF//, public e_DerivedTypeHelper<8>
{
	TYPE_FUNC(8)
	float m_fdrInt, m_fdrExt, m_eta, m_invEta2;
	e_Texture m_diffuseReflectance;
	e_Texture m_specularReflectance;
	float m_specularSamplingWeight;
	bool m_nonlinear;
	plastic()
		: BSDF(EBSDFType(EDeltaReflection | EDiffuseReflection), 0, &m_diffuseReflectance, &m_specularReflectance)
	{
	}
	plastic(float eta, const e_Texture& d, bool nonlinear = false)
		: m_diffuseReflectance(d), m_nonlinear(nonlinear), BSDF(EBSDFType(EDeltaReflection | EDiffuseReflection), 0, &m_diffuseReflectance, &m_specularReflectance)
	{
		m_specularReflectance = CreateTexture(Spectrum(1.0f));
		m_eta = eta;
		m_invEta2 = 1.0f / (eta * eta);
		m_fdrInt = fresnelDiffuseReflectance(1/m_eta);
		m_fdrExt = fresnelDiffuseReflectance(m_eta);
		float dAvg = m_diffuseReflectance.Average().getLuminance(), sAvg = m_specularReflectance.Average().getLuminance();
		m_specularSamplingWeight = sAvg / (dAvg + sAvg);
	}
	plastic(float eta, const e_Texture& d, const e_Texture& r, bool nonlinear = false)
		: m_diffuseReflectance(d), m_nonlinear(nonlinear), BSDF(EBSDFType(EDeltaReflection | EDiffuseReflection), 0, &m_diffuseReflectance, &m_specularReflectance)
	{
		m_specularReflectance = r;
		m_eta = eta;
		m_invEta2 = 1.0f / (eta * eta);
		m_fdrInt = fresnelDiffuseReflectance(1/m_eta);
		m_fdrExt = fresnelDiffuseReflectance(m_eta);
		float dAvg = m_diffuseReflectance.Average().getLuminance(), sAvg = m_specularReflectance.Average().getLuminance();
		m_specularSamplingWeight = sAvg / (dAvg + sAvg);
	}
	virtual void Update()
	{
		m_invEta2 = 1.0f / (m_eta * m_eta);
		m_fdrInt = fresnelDiffuseReflectance(1/m_eta);
		m_fdrExt = fresnelDiffuseReflectance(m_eta);
		float dAvg = m_diffuseReflectance.Average().getLuminance(), sAvg = m_specularReflectance.Average().getLuminance();
		m_specularSamplingWeight = sAvg / (dAvg + sAvg);
	}
	CUDA_FUNC_IN Vec3f reflect(const Vec3f &wi) const {
		return Vec3f(-wi.x, -wi.y, wi.z);
	}
	CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const Vec2f &sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
private:
	float fresnelDiffuseReflectance(float eta) const {
		/* Fast mode: the following code approximates the
		 * diffuse Frensel reflectance for the eta<1 and
		 * eta>1 cases. An evalution of the accuracy led
		 * to the following scheme, which cherry-picks
		 * fits from two papers where they are best.
		 */
		if (eta < 1) {
			/* Fit by Egan and Hilgeman (1973). Works
			   reasonably well for "normal" IOR values (<2).

			   Max rel. error in 1.0 - 1.5 : 0.1%
			   Max rel. error in 1.5 - 2   : 0.6%
			   Max rel. error in 2.0 - 5   : 9.5%
			*/
			return -1.4399f * (eta * eta)
				  + 0.7099f * eta
				  + 0.6681f
				  + 0.0636f / eta;
		} else {
			/* Fit by d'Eon and Irving (2011)
			 *
			 * Maintains a good accuracy even for
			 * unrealistic IOR values.
			 *
			 * Max rel. error in 1.0 - 2.0   : 0.1%
			 * Max rel. error in 2.0 - 10.0  : 0.2%
			 */
			float invEta = 1.0f / eta,
				  invEta2 = invEta*invEta,
				  invEta3 = invEta2*invEta,
				  invEta4 = invEta3*invEta,
				  invEta5 = invEta4*invEta;

			return 0.919317f - 3.4793f * invEta
				 + 6.75335f * invEta2
				 - 7.80989f * invEta3
				 + 4.98554f * invEta4
				 - 1.36881f * invEta5;
		}
	}
};

struct roughplastic : public BSDF//, public e_DerivedTypeHelper<9>
{
	TYPE_FUNC(9)
	MicrofacetDistribution m_distribution;
	e_Texture m_diffuseReflectance;
	e_Texture m_specularReflectance;
	e_Texture m_alpha;
	float m_eta, m_invEta2;
	float m_specularSamplingWeight;
	bool m_nonlinear;
	roughplastic()
		: BSDF(EBSDFType(EGlossyReflection | EDiffuseReflection), 0, &m_diffuseReflectance, &m_specularReflectance, &m_alpha)
	{
	}
	roughplastic(MicrofacetDistribution::EType type, float eta, e_Texture& alpha, e_Texture& diffuse, bool nonlinear = false)
		: BSDF(EBSDFType(EGlossyReflection | EDiffuseReflection), 0, &m_diffuseReflectance, &m_specularReflectance, &m_alpha), m_eta(eta), m_invEta2(1.0f / (eta * eta)), m_alpha(alpha), m_diffuseReflectance(diffuse), m_nonlinear(nonlinear)
	{
		m_distribution.m_type = type;
		m_specularReflectance = CreateTexture(Spectrum(1.0f));
		float dAvg = m_diffuseReflectance.Average().getLuminance(), sAvg = m_specularReflectance.Average().getLuminance();
		m_specularSamplingWeight = sAvg / (dAvg + sAvg);
	}
	roughplastic(MicrofacetDistribution::EType type, float eta, e_Texture& alpha, e_Texture& diffuse, e_Texture& specular, bool nonlinear = false)
		: BSDF(EBSDFType(EGlossyReflection | EDiffuseReflection), 0, &m_diffuseReflectance, &m_specularReflectance, &m_alpha), m_eta(eta), m_invEta2(1.0f / (eta * eta)), m_alpha(alpha), m_diffuseReflectance(diffuse), m_nonlinear(nonlinear)
	{
		m_distribution.m_type = type;
		m_specularReflectance = specular;
		float dAvg = m_diffuseReflectance.Average().getLuminance(), sAvg = m_specularReflectance.Average().getLuminance();
		m_specularSamplingWeight = sAvg / (dAvg + sAvg);
	}
	virtual void Update()
	{
		m_invEta2 = 1.0f / (m_eta * m_eta);
		float dAvg = m_diffuseReflectance.Average().getLuminance(), sAvg = m_specularReflectance.Average().getLuminance();
		m_specularSamplingWeight = sAvg / (dAvg + sAvg);
	}
	/// Helper function: reflect \c wi with respect to a given surface normal
	CUDA_FUNC_IN Vec3f reflect(const Vec3f &wi, const Vec3f &m) const {
		return 2 * dot(wi, m) * m - wi;
	}
	CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const Vec2f &sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
};

struct phong : public BSDF//, public e_DerivedTypeHelper<10>
{
	TYPE_FUNC(10)
	e_Texture m_diffuseReflectance;
	e_Texture m_specularReflectance;
	e_Texture m_exponent;
	float m_specularSamplingWeight;
	phong()
		: BSDF(EBSDFType(EGlossyReflection | EDiffuseReflection), 0, &m_diffuseReflectance, &m_specularReflectance, &m_exponent)
	{
	}
	phong(const e_Texture& d, const e_Texture& s, const e_Texture& e)
		: m_diffuseReflectance(d), m_specularReflectance(s), m_exponent(e), BSDF(EBSDFType(EGlossyReflection | EDiffuseReflection), 0, &m_diffuseReflectance, &m_specularReflectance, &m_exponent)
	{
		float dAvg = m_diffuseReflectance.Average().getLuminance(), sAvg = m_specularReflectance.Average().getLuminance();
		m_specularSamplingWeight = sAvg / (dAvg + sAvg);
	}
	virtual void Update()
	{
		float dAvg = m_diffuseReflectance.Average().getLuminance(), sAvg = m_specularReflectance.Average().getLuminance();
		m_specularSamplingWeight = sAvg / (dAvg + sAvg);
	}
	CUDA_FUNC_IN Vec3f reflect(const Vec3f &wi) const {
		return Vec3f(-wi.x, -wi.y, wi.z);
	}	
	CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &_pdf, const Vec2f& _sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
};

struct ward : public BSDF//, public e_DerivedTypeHelper<11>
{
	TYPE_FUNC(11)
	enum EModelVariant {
		/// The original Ward model
		EWard = 0,
		/// Ward model with correction by Arne Duer
		EWardDuer = 1,
		/// Energy-balanced Ward model
		EBalanced = 2
	};
	EModelVariant m_modelVariant;
	float m_specularSamplingWeight;
	e_Texture m_diffuseReflectance;
	e_Texture m_specularReflectance;
	e_Texture m_alphaU;
	e_Texture m_alphaV;
	ward()
		: BSDF(EBSDFType(EGlossyReflection | EDiffuseReflection), 0, &m_diffuseReflectance, &m_specularReflectance, &m_alphaU, &m_alphaV)
	{
	}
	ward(EModelVariant type, const e_Texture& d, const e_Texture& s, const e_Texture& u, const e_Texture& v)
		: m_modelVariant(type), m_diffuseReflectance(d), m_specularReflectance(s), m_alphaU(u), m_alphaV(v), BSDF(EBSDFType(EGlossyReflection | EDiffuseReflection), 0, &m_diffuseReflectance, &m_specularReflectance, &m_alphaU, &m_alphaV)
	{
		float dAvg = m_diffuseReflectance.Average().getLuminance(), sAvg = m_specularReflectance.Average().getLuminance();
		m_specularSamplingWeight = sAvg / (dAvg + sAvg);
	}
	virtual void Update()
	{
		float dAvg = m_diffuseReflectance.Average().getLuminance(), sAvg = m_specularReflectance.Average().getLuminance();
		m_specularSamplingWeight = sAvg / (dAvg + sAvg);
	}
	CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &_pdf, const Vec2f &_sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
};

struct hk : public BSDF//, public e_DerivedTypeHelper<12>
{
	TYPE_FUNC(12)
	e_Texture m_sigmaS;
	e_Texture m_sigmaA;
	e_PhaseFunction m_phase;
	float m_thickness;
	hk()
		: BSDF(EBSDFType(EGlossyReflection | EGlossyTransmission | EDeltaTransmission), 0, &m_sigmaS, &m_sigmaA)
	{
	}
	hk(const e_Texture& ss, const e_Texture& sa, e_PhaseFunction& phase, float thickness)
		: m_sigmaS(ss), m_sigmaA(sa), m_phase(phase), m_thickness(thickness), BSDF(EBSDFType(EGlossyReflection | EGlossyTransmission | EDeltaTransmission), 0, &m_sigmaS, &m_sigmaA)
	{
	}
	CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &_pdf, const Vec2f &_sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
};