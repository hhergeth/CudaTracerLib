#pragma once

#define diffuse_TYPE 1
struct diffuse : public BSDF
{
	e_KernelTexture	m_reflectance;
	diffuse()
		: BSDF(EDiffuseReflection)
	{
	}
	diffuse(const e_KernelTexture& d)
		: m_reflectance(d), BSDF(EDiffuseReflection)
	{
	}
	CUDA_FUNC_IN Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const
	{
		if (!(bRec.typeMask & EDiffuseReflection) || Frame::cosTheta(bRec.wi) <= 0)
			return make_float3(0.0f);

		bRec.wo = Warp::squareToCosineHemisphere(sample);
		bRec.eta = 1.0f;
		bRec.sampledType = EDiffuseReflection;
		pdf = Warp::squareToCosineHemispherePdf(bRec.wo);
		return m_reflectance.Evaluate(bRec.map);
	}
	CUDA_FUNC_IN Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const
	{
		if (!(bRec.typeMask & EDiffuseReflection) || measure != ESolidAngle
			|| Frame::cosTheta(bRec.wi) <= 0
			|| Frame::cosTheta(bRec.wo) <= 0)
			return make_float3(0.0f);

		return m_reflectance.Evaluate(bRec.map)	* (INV_PI * Frame::cosTheta(bRec.wo));
	}
	CUDA_FUNC_IN float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const
	{
		if (!(bRec.typeMask & EDiffuseReflection) || measure != ESolidAngle
			|| Frame::cosTheta(bRec.wi) <= 0
			|| Frame::cosTheta(bRec.wo) <= 0)
			return 0.0f;

		return Warp::squareToCosineHemispherePdf(bRec.wo);
	}
	CUDA_FUNC_IN Spectrum getDiffuseReflectance(BSDFSamplingRecord &bRec) const
	{
		return m_reflectance.Evaluate(bRec.map);
	}
	template<typename T> void LoadTextures(T callback)
	{
		m_reflectance.LoadTextures(callback);
	}
	TYPE_FUNC(diffuse)
};

#define roughdiffuse_TYPE 2
struct roughdiffuse : public BSDF
{
	e_KernelTexture	m_reflectance;
	e_KernelTexture	m_alpha;
	bool m_useFastApprox;
	roughdiffuse()
		: BSDF(EDiffuseReflection)
	{
	}
	roughdiffuse(const e_KernelTexture& r, const e_KernelTexture& a)
		: m_reflectance(r), m_alpha(a), BSDF(EDiffuseReflection)
	{
	}
	CUDA_FUNC_IN Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const
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
	CUDA_FUNC_IN Spectrum getDiffuseReflectance(BSDFSamplingRecord &bRec) const
	{
		return m_reflectance.Evaluate(bRec.map);
	}
	template<typename T> void LoadTextures(T callback)
	{
		m_reflectance.LoadTextures(callback);
		m_alpha.LoadTextures(callback);
	}
	TYPE_FUNC(roughdiffuse)
};

#define dielectric_TYPE 3
struct dielectric : public BSDF
{
	float m_eta, m_invEta;
	e_KernelTexture m_specularTransmittance;
	e_KernelTexture m_specularReflectance;
	dielectric()
		: BSDF(EDeltaReflection | EDeltaTransmission)
	{
	}
	dielectric(float e)
		: BSDF(EDeltaReflection | EDeltaTransmission)
	{
		m_eta = e;
		m_invEta = 1.0f / e;
		m_specularTransmittance = CreateTexture(0, Spectrum(1.0f));
		m_specularReflectance = CreateTexture(0, Spectrum(1.0f));
	}
	dielectric(float e, const Spectrum& r, const Spectrum& t)
		: BSDF(EDeltaReflection | EDeltaTransmission)
	{
		m_eta = e;
		m_invEta = 1.0f / e;
		m_specularTransmittance = CreateTexture(0, t);
		m_specularReflectance = CreateTexture(0, r);
	}
	dielectric(float e, const e_KernelTexture& r, const e_KernelTexture& t)
		: BSDF(EDeltaReflection | EDeltaTransmission)
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
	CUDA_FUNC_IN float3 reflect(const float3 &wi) const {
		return make_float3(-wi.x, -wi.y, wi.z);
	}
	/// Refraction in local coordinates
	CUDA_FUNC_IN float3 refract(const float3 &wi, float cosThetaT) const {
		float scale = -(cosThetaT < 0 ? m_invEta : m_eta);
		return make_float3(scale*wi.x, scale*wi.y, cosThetaT);
	}
	CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	STD_DIFFUSE_REFLECTANCE
	template<typename T> void LoadTextures(T callback)
	{
		m_specularTransmittance.LoadTextures(callback);
		m_specularReflectance.LoadTextures(callback);
	}
	TYPE_FUNC(dielectric)
};

#define thindielectric_TYPE 4
struct thindielectric : public BSDF
{
	float m_eta;
	e_KernelTexture m_specularTransmittance;
	e_KernelTexture m_specularReflectance;
	thindielectric()
		: BSDF(EDeltaReflection | EDeltaTransmission)
	{
	}
	thindielectric(float e)
		: BSDF(EDeltaReflection | EDeltaTransmission)
	{
		m_eta = e;
		m_specularTransmittance = CreateTexture(0, Spectrum(1.0f));
		m_specularReflectance = CreateTexture(0, Spectrum(1.0f));
	}
	thindielectric(float e, const e_KernelTexture& r, const e_KernelTexture& t)
		: BSDF(EDeltaReflection | EDeltaTransmission)
	{
		m_eta = e;
		m_specularTransmittance = t;
		m_specularReflectance = r;
	}
	CUDA_FUNC_IN float3 transmit(const float3 &wi) const {
		return -1.0f * wi;
	}
	/// Reflection in local coordinates
	CUDA_FUNC_IN float3 reflect(const float3 &wi) const {
		return make_float3(-wi.x, -wi.y, wi.z);
	}
	CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	STD_DIFFUSE_REFLECTANCE
	template<typename T> void LoadTextures(T callback)
	{
		m_specularTransmittance.LoadTextures(callback);
		m_specularReflectance.LoadTextures(callback);
	}
	TYPE_FUNC(thindielectric)
};

#define roughdielectric_TYPE 5
struct roughdielectric : public BSDF
{
	MicrofacetDistribution m_distribution;
	e_KernelTexture m_specularTransmittance;
	e_KernelTexture  m_specularReflectance;
	e_KernelTexture m_alphaU, m_alphaV;
	float m_eta, m_invEta;
	roughdielectric()
		: BSDF(EGlossyReflection | EGlossyTransmission)
	{
	}
	roughdielectric(MicrofacetDistribution::EType type, float eta, const e_KernelTexture& u, const e_KernelTexture& v)
		: m_alphaU(u), m_alphaV(v), BSDF(EGlossyReflection | EGlossyTransmission)
	{
		m_distribution.m_type = type;
		m_specularTransmittance = CreateTexture(0, Spectrum(1.0f));
		m_specularReflectance = CreateTexture(0, Spectrum(1.0f));
		m_eta = eta;
		m_invEta = 1.0f / eta;
	}
	roughdielectric(MicrofacetDistribution::EType type, float eta, const e_KernelTexture& u, const e_KernelTexture& v, const e_KernelTexture& r, const e_KernelTexture& t)
		: m_alphaU(u), m_alphaV(v), BSDF(EGlossyReflection | EGlossyTransmission)
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
	CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &_sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	STD_DIFFUSE_REFLECTANCE
	template<typename T> void LoadTextures(T callback)
	{
		m_specularTransmittance.LoadTextures(callback);
		m_specularReflectance.LoadTextures(callback);
		m_alphaU.LoadTextures(callback);
		m_alphaV.LoadTextures(callback);
	}
	TYPE_FUNC(roughdielectric)
};

#define conductor_TYPE 6
struct conductor : public BSDF
{
	e_KernelTexture m_specularReflectance;
	Spectrum m_eta;
	Spectrum m_k;
	conductor()
		: BSDF(EDeltaReflection)
	{
	}
	conductor(const Spectrum& eta, const Spectrum& k)
		: BSDF(EDeltaReflection)
	{
		m_specularReflectance = CreateTexture(0, Spectrum(1.0f));
		m_eta = eta;
		m_k = k;
	}
	conductor(const Spectrum& eta, const Spectrum& k, const e_KernelTexture& r)
		: BSDF(EDeltaReflection)
	{
		m_specularReflectance = r;
		m_eta = eta;
		m_k = k;
	}
	/// Reflection in local coordinates
	CUDA_FUNC_IN float3 reflect(const float3 &wi) const {
		return make_float3(-wi.x, -wi.y, wi.z);
	}
	CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	STD_DIFFUSE_REFLECTANCE
	template<typename T> void LoadTextures(T callback)
	{
		m_specularReflectance.LoadTextures(callback);
	}
	TYPE_FUNC(conductor)
};

#define roughconductor_TYPE 7
struct roughconductor : public BSDF
{	
	MicrofacetDistribution m_distribution;
	e_KernelTexture m_specularReflectance;
	e_KernelTexture m_alphaU, m_alphaV;
	Spectrum m_eta, m_k;
	roughconductor()
		: BSDF(EGlossyReflection)
	{
	}
	roughconductor(MicrofacetDistribution::EType type, const Spectrum& eta, const Spectrum& k, const e_KernelTexture& u, const e_KernelTexture& v)
		: m_alphaU(u), m_alphaV(v), BSDF(EGlossyReflection)
	{
		m_specularReflectance = CreateTexture(0, Spectrum(1.0f));
		m_eta = eta;
		m_k = k;
		m_distribution.m_type = type;
	}
	roughconductor(MicrofacetDistribution::EType type, const Spectrum& eta, const Spectrum& k, const e_KernelTexture& u, const e_KernelTexture& v, const e_KernelTexture& r)
		: m_alphaU(u), m_alphaV(v), BSDF(EGlossyReflection)
	{
		m_specularReflectance = r;
		m_eta = eta;
		m_k = k;
		m_distribution.m_type = type;
	}
	CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	STD_DIFFUSE_REFLECTANCE
	template<typename T> void LoadTextures(T callback)
	{
		m_specularReflectance.LoadTextures(callback);
		m_alphaU.LoadTextures(callback);
		m_alphaV.LoadTextures(callback);
	}
	TYPE_FUNC(roughconductor)
};

#define plastic_TYPE 8
struct plastic : public BSDF
{
	float m_fdrInt, m_fdrExt, m_eta, m_invEta2;
	e_KernelTexture m_diffuseReflectance;
	e_KernelTexture m_specularReflectance;
	float m_specularSamplingWeight;
	bool m_nonlinear;
	plastic()
		: BSDF(EDeltaReflection | EDiffuseReflection)
	{
	}
	plastic(float eta, const e_KernelTexture& d, bool nonlinear = false)
		: m_diffuseReflectance(d), m_nonlinear(nonlinear), BSDF(EDeltaReflection | EDiffuseReflection)
	{
		m_specularReflectance = CreateTexture(0, Spectrum(1.0f));
		m_eta = eta;
		m_invEta2 = 1.0f / (eta * eta);
		m_fdrInt = fresnelDiffuseReflectance(1/m_eta);
		m_fdrExt = fresnelDiffuseReflectance(m_eta);
		float dAvg = m_diffuseReflectance.Average().getLuminance(), sAvg = m_specularReflectance.Average().getLuminance();
		m_specularSamplingWeight = sAvg / (dAvg + sAvg);
	}
	plastic(float eta, const e_KernelTexture& d, const e_KernelTexture& r, bool nonlinear = false)
		: m_diffuseReflectance(d), m_nonlinear(nonlinear), BSDF(EDeltaReflection | EDiffuseReflection)
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
	CUDA_FUNC_IN float3 reflect(const float3 &wi) const {
		return make_float3(-wi.x, -wi.y, wi.z);
	}
	CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_FUNC_IN Spectrum getDiffuseReflectance(BSDFSamplingRecord &bRec) const
	{
		return m_diffuseReflectance.Evaluate(bRec.map) * (1.0f - m_fdrExt);
	}
	template<typename T> void LoadTextures(T callback)
	{
		m_diffuseReflectance.LoadTextures(callback);
		m_specularReflectance.LoadTextures(callback);
	}
	TYPE_FUNC(plastic)
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

#define roughplastic_TYPE 9
struct roughplastic : public BSDF
{
	MicrofacetDistribution m_distribution;
	e_KernelTexture m_diffuseReflectance;
	e_KernelTexture m_specularReflectance;
	e_KernelTexture m_alpha;
	float m_eta, m_invEta2;
	float m_specularSamplingWeight;
	bool m_nonlinear;
	roughplastic()
		: BSDF(EGlossyReflection | EDiffuseReflection)
	{
	}
	roughplastic(MicrofacetDistribution::EType type, float eta, e_KernelTexture& alpha, e_KernelTexture& diffuse, bool nonlinear = false)
		: BSDF(EGlossyReflection | EDiffuseReflection), m_eta(eta), m_invEta2(1.0f / (eta * eta)), m_alpha(alpha), m_diffuseReflectance(diffuse), m_nonlinear(nonlinear)
	{
		m_distribution.m_type = type;
		m_specularReflectance = CreateTexture(0, Spectrum(1.0f));
		float dAvg = m_diffuseReflectance.Average().getLuminance(), sAvg = m_specularReflectance.Average().getLuminance();
		m_specularSamplingWeight = sAvg / (dAvg + sAvg);
	}
	roughplastic(MicrofacetDistribution::EType type, float eta, e_KernelTexture& alpha, e_KernelTexture& diffuse, e_KernelTexture& specular, bool nonlinear = false)
		: BSDF(EGlossyReflection | EDiffuseReflection), m_eta(eta), m_invEta2(1.0f / (eta * eta)), m_alpha(alpha), m_diffuseReflectance(diffuse), m_nonlinear(nonlinear)
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
	CUDA_FUNC_IN float3 reflect(const float3 &wi, const float3 &m) const {
		return 2 * dot(wi, m) * m - wi;
	}
	CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_FUNC_IN Spectrum getDiffuseReflectance(BSDFSamplingRecord &bRec) const
	{
		float alpha = m_alpha.Evaluate(bRec.map).average();
		float Ftr = e_RoughTransmittanceManager::EvaluateDiffuse(m_distribution.m_type, alpha, m_eta);

		return m_diffuseReflectance.Evaluate(bRec.map) * Ftr;
	}
	template<typename T> void LoadTextures(T callback)
	{
		m_diffuseReflectance.LoadTextures(callback);
		m_specularReflectance.LoadTextures(callback);
		m_alpha.LoadTextures(callback);
	}
	TYPE_FUNC(roughplastic)
};

#define phong_TYPE 10
struct phong : public BSDF
{
	e_KernelTexture m_diffuseReflectance;
	e_KernelTexture m_specularReflectance;
	e_KernelTexture m_exponent;
	float m_specularSamplingWeight;
	phong()
		: BSDF(EGlossyReflection | EDiffuseReflection)
	{
	}
	phong(const e_KernelTexture& d, const e_KernelTexture& s, const e_KernelTexture& e)
		: m_diffuseReflectance(d), m_specularReflectance(s), m_exponent(e), BSDF(EGlossyReflection | EDiffuseReflection)
	{
		float dAvg = m_diffuseReflectance.Average().getLuminance(), sAvg = m_specularReflectance.Average().getLuminance();
		m_specularSamplingWeight = sAvg / (dAvg + sAvg);
	}
	virtual void Update()
	{
		float dAvg = m_diffuseReflectance.Average().getLuminance(), sAvg = m_specularReflectance.Average().getLuminance();
		m_specularSamplingWeight = sAvg / (dAvg + sAvg);
	}
	CUDA_FUNC_IN float3 reflect(const float3 &wi) const {
		return make_float3(-wi.x, -wi.y, wi.z);
	}	
	CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &_pdf, const float2& _sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_FUNC_IN Spectrum getDiffuseReflectance(BSDFSamplingRecord &bRec) const
	{
		return m_diffuseReflectance.Evaluate(bRec.map);
	}
	template<typename T> void LoadTextures(T callback)
	{
		m_diffuseReflectance.LoadTextures(callback);
		m_specularReflectance.LoadTextures(callback);
		m_exponent.LoadTextures(callback);
	}
	TYPE_FUNC(phong)
};

#define ward_TYPE 11
struct ward : public BSDF
{
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
	e_KernelTexture m_diffuseReflectance;
	e_KernelTexture m_specularReflectance;
	e_KernelTexture m_alphaU;
	e_KernelTexture m_alphaV;
	ward()
		: BSDF(EGlossyReflection | EDiffuseReflection)
	{
	}
	ward(EModelVariant type, const e_KernelTexture& d, const e_KernelTexture& s, const e_KernelTexture& u, const e_KernelTexture& v)
		: m_modelVariant(type), m_diffuseReflectance(d), m_specularReflectance(s), m_alphaU(u), m_alphaV(v), BSDF(EGlossyReflection | EDiffuseReflection)
	{
		float dAvg = m_diffuseReflectance.Average().getLuminance(), sAvg = m_specularReflectance.Average().getLuminance();
		m_specularSamplingWeight = sAvg / (dAvg + sAvg);
	}
	virtual void Update()
	{
		float dAvg = m_diffuseReflectance.Average().getLuminance(), sAvg = m_specularReflectance.Average().getLuminance();
		m_specularSamplingWeight = sAvg / (dAvg + sAvg);
	}
	CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &_pdf, const float2 &_sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_FUNC_IN Spectrum getDiffuseReflectance(BSDFSamplingRecord &bRec) const
	{
		return m_diffuseReflectance.Evaluate(bRec.map);
	}
	template<typename T> void LoadTextures(T callback)
	{
		m_diffuseReflectance.LoadTextures(callback);
		m_specularReflectance.LoadTextures(callback);
		m_alphaU.LoadTextures(callback);
		m_alphaV.LoadTextures(callback);
	}
	TYPE_FUNC(ward)
};

#define hk_TYPE 12
struct hk : public BSDF
{
	e_KernelTexture m_sigmaS;
	e_KernelTexture m_sigmaA;
	e_PhaseFunction m_phase;
	float m_thickness;
	hk()
		: BSDF(EGlossyReflection | EGlossyTransmission | EDeltaTransmission)
	{
	}
	hk(const e_KernelTexture& ss, const e_KernelTexture& sa, e_PhaseFunction& phase, float thickness)
		: m_sigmaS(ss), m_sigmaA(sa), m_phase(phase), m_thickness(thickness), BSDF(EGlossyReflection | EGlossyTransmission | EDeltaTransmission)
	{
	}
	CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &_pdf, const float2 &_sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_FUNC_IN Spectrum getDiffuseReflectance(BSDFSamplingRecord &bRec) const
	{
		Spectrum sigmaA = m_sigmaA.Evaluate(bRec.map),
				 sigmaS = m_sigmaS.Evaluate(bRec.map),
				 sigmaT = sigmaA + sigmaS,
				 albedo;
		for (int i = 0; i < SPECTRUM_SAMPLES; i++)
			albedo[i] = sigmaT[i] > 0 ? (sigmaS[i]/sigmaT[i]) : (float) 0;
		return albedo;
	}
	template<typename T> void LoadTextures(T callback)
	{
		m_sigmaS.LoadTextures(callback);
		m_sigmaA.LoadTextures(callback);
	}
	TYPE_FUNC(hk)
};