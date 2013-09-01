#pragma once

#define diffuse_TYPE 1
struct diffuse : public BSDF
{
	e_KernelTexture<float3>	m_reflectance;
	diffuse()
		: BSDF(EDiffuseReflection)
	{
	}
	diffuse(const e_KernelTexture<float3>& d)
		: m_reflectance(d), BSDF(EDiffuseReflection)
	{
	}
	CUDA_FUNC_IN float3 sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const
	{
		if (!(bRec.typeMask & EDiffuseReflection) || Frame::cosTheta(bRec.wi) <= 0)
			return make_float3(0.0f);

		bRec.wo = Warp::squareToCosineHemisphere(sample);
		bRec.eta = 1.0f;
		bRec.sampledComponent = 0;
		bRec.sampledType = EDiffuseReflection;
		pdf = Warp::squareToCosineHemispherePdf(bRec.wo);
		return m_reflectance.Evaluate(bRec.map);
	}
	CUDA_FUNC_IN float3 f(const BSDFSamplingRecord &bRec, EMeasure measure) const
	{
		if (!(bRec.typeMask & EDiffuseReflection) || measure != ESolidAngle
			|| Frame::cosTheta(bRec.wi) <= 0
			|| Frame::cosTheta(bRec.wo) <= 0)
			return make_float3(0.0f);

		return m_reflectance.Evaluate(bRec.map)
			* (INV_PI * Frame::cosTheta(bRec.wo));
	}
	CUDA_FUNC_IN float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const
	{
		if (!(bRec.typeMask & EDiffuseReflection) || measure != ESolidAngle
			|| Frame::cosTheta(bRec.wi) <= 0
			|| Frame::cosTheta(bRec.wo) <= 0)
			return 0.0f;

		return Warp::squareToCosineHemispherePdf(bRec.wo);
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
	e_KernelTexture<float3>	m_reflectance;
	e_KernelTexture<float>	m_alpha;
	bool m_useFastApprox;
	roughdiffuse()
		: BSDF(EDiffuseReflection)
	{
	}
	roughdiffuse(const e_KernelTexture<float3>& r, const e_KernelTexture<float>& a)
		: m_reflectance(r), m_alpha(a), BSDF(EDiffuseReflection)
	{
	}
	CUDA_FUNC_IN float3 sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const
	{
		bRec.wo = Warp::squareToCosineHemisphere(sample);
		bRec.eta = 1.0f;
		bRec.sampledComponent = 0;
		bRec.sampledType = EGlossyReflection;
		pdf = Warp::squareToCosineHemispherePdf(bRec.wo);
		return f(bRec, ESolidAngle) / pdf;
	}
	CUDA_DEVICE CUDA_HOST float3 f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_FUNC_IN float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const
	{
		if (!(bRec.typeMask & EGlossyReflection) || measure != ESolidAngle
			|| Frame::cosTheta(bRec.wi) <= 0
			|| Frame::cosTheta(bRec.wo) <= 0)
			return 0.0f;

		return Warp::squareToCosineHemispherePdf(bRec.wo);
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
	e_KernelTexture<float3> m_specularTransmittance;
	e_KernelTexture<float3> m_specularReflectance;
	dielectric()
		: BSDF(EDeltaReflection | EDeltaTransmission)
	{
	}
	dielectric(float e)
		: BSDF(EDeltaReflection | EDeltaTransmission)
	{
		m_eta = e;
		m_invEta = 1.0f / e;
		m_specularTransmittance = CreateTexture(0, make_float3(1));
		m_specularReflectance = CreateTexture(0, make_float3(1));
	}
	dielectric(float e, const float3& r, const float3& t)
		: BSDF(EDeltaReflection | EDeltaTransmission)
	{
		m_eta = e;
		m_invEta = 1.0f / e;
		m_specularTransmittance = CreateTexture(0, t);
		m_specularReflectance = CreateTexture(0, r);
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
	CUDA_DEVICE CUDA_HOST float3 sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const;
	CUDA_DEVICE CUDA_HOST float3 f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
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
	e_KernelTexture<float3> m_specularTransmittance;
	e_KernelTexture<float3> m_specularReflectance;
	thindielectric()
		: BSDF(EDeltaReflection | EDeltaTransmission)
	{
	}
	thindielectric(float e)
		: BSDF(EDeltaReflection | EDeltaTransmission)
	{
		m_eta = e;
		m_specularTransmittance = CreateTexture(0, make_float3(1));
		m_specularReflectance = CreateTexture(0, make_float3(1));
	}
	CUDA_FUNC_IN float3 transmit(const float3 &wi) const {
		return -1.0f * wi;
	}
	/// Reflection in local coordinates
	CUDA_FUNC_IN float3 reflect(const float3 &wi) const {
		return make_float3(-wi.x, -wi.y, wi.z);
	}
	CUDA_DEVICE CUDA_HOST float3 sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const;
	CUDA_DEVICE CUDA_HOST float3 f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
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
	e_KernelTexture<float3> m_specularTransmittance;
	e_KernelTexture<float3>  m_specularReflectance;
	e_KernelTexture<float> m_alphaU, m_alphaV;
	float m_eta, m_invEta;
	roughdielectric()
		: BSDF(EGlossyReflection | EGlossyTransmission)
	{
	}
	roughdielectric(MicrofacetDistribution::EType type, float eta, const e_KernelTexture<float>& u, const e_KernelTexture<float>& v)
		: m_alphaU(u), m_alphaV(v), BSDF(EGlossyReflection | EGlossyTransmission)
	{
		m_distribution.m_type = type;
		m_specularTransmittance = CreateTexture(0, make_float3(1));
		m_specularReflectance = CreateTexture(0, make_float3(1));
		m_eta = eta;
		m_invEta = 1.0f / eta;
	}
	CUDA_DEVICE CUDA_HOST float3 sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &_sample) const;
	CUDA_DEVICE CUDA_HOST float3 f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
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
	e_KernelTexture<float3> m_specularReflectance;
	float3 m_eta;
	float3 m_k;
	conductor()
		: BSDF(EDeltaReflection)
	{
	}
	conductor(const float3& eta, const float3& k)
		: BSDF(EDeltaReflection)
	{
		m_specularReflectance = CreateTexture(0, make_float3(1));
		m_eta = eta;
		m_k = k;
	}
	/// Reflection in local coordinates
	CUDA_FUNC_IN float3 reflect(const float3 &wi) const {
		return make_float3(-wi.x, -wi.y, wi.z);
	}
	CUDA_DEVICE CUDA_HOST float3 sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const;
	CUDA_DEVICE CUDA_HOST float3 f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
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
	e_KernelTexture<float3> m_specularReflectance;
	e_KernelTexture<float> m_alphaU, m_alphaV;
	float3 m_eta, m_k;
	roughconductor()
		: BSDF(EGlossyReflection)
	{
	}
	roughconductor(MicrofacetDistribution::EType type, const float3& eta, const float3& k, const e_KernelTexture<float>& u, const e_KernelTexture<float>& v)
		: m_alphaU(u), m_alphaV(v), BSDF(EGlossyReflection)
	{
		m_specularReflectance = CreateTexture(0, make_float3(1));
		m_eta = eta;
		m_k = k;
		m_distribution.m_type = type;
	}
	CUDA_DEVICE CUDA_HOST float3 sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const;
	CUDA_DEVICE CUDA_HOST float3 f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
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
	e_KernelTexture<float3> m_diffuseReflectance;
	e_KernelTexture<float3> m_specularReflectance;
	float m_specularSamplingWeight;
	bool m_nonlinear;
	plastic()
		: BSDF(EDeltaReflection | EDiffuseReflection)
	{
	}
	plastic(float eta, const e_KernelTexture<float3>& d, bool nonlinear = false)
		: m_diffuseReflectance(d), m_nonlinear(nonlinear), BSDF(EDeltaReflection | EDiffuseReflection)
	{
		m_specularReflectance = CreateTexture(0, make_float3(1));
		m_eta = eta;
		m_invEta2 = 1.0f / (eta * eta);
		m_fdrInt = fresnelDiffuseReflectance(1/m_eta);
		m_fdrExt = fresnelDiffuseReflectance(m_eta);
		MapParameters mp(make_float3(0), make_float2(0), Frame(make_float3(0,1,0)));
		float dAvg = y(m_diffuseReflectance.Evaluate(mp)), sAvg = y(m_specularReflectance.Evaluate(mp));
		m_specularSamplingWeight = sAvg / (dAvg + sAvg);
	}
	CUDA_FUNC_IN float3 reflect(const float3 &wi) const {
		return make_float3(-wi.x, -wi.y, wi.z);
	}
	CUDA_DEVICE CUDA_HOST float3 sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const;
	CUDA_DEVICE CUDA_HOST float3 f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
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
	return 0.0f;
}
};

#define phong_TYPE 9
struct phong : public BSDF
{
	e_KernelTexture<float3> m_diffuseReflectance;
	e_KernelTexture<float3> m_specularReflectance;
	e_KernelTexture<float> m_exponent;
	float m_specularSamplingWeight;
	phong()
		: BSDF(EGlossyReflection | EDiffuseReflection)
	{
	}
	phong(const e_KernelTexture<float3>& d, const e_KernelTexture<float3>& s, const e_KernelTexture<float>& e)
		: m_diffuseReflectance(d), m_specularReflectance(s), m_exponent(e), BSDF(EGlossyReflection | EDiffuseReflection)
	{
		MapParameters mp(make_float3(0), make_float2(0), Frame(make_float3(0,1,0)));
		float dAvg = y(m_diffuseReflectance.Evaluate(mp)), sAvg = y(m_specularReflectance.Evaluate(mp));
		m_specularSamplingWeight = sAvg / (dAvg + sAvg);
	}
	CUDA_FUNC_IN float3 reflect(const float3 &wi) const {
		return make_float3(-wi.x, -wi.y, wi.z);
	}	
	CUDA_DEVICE CUDA_HOST float3 sample(BSDFSamplingRecord &bRec, float &_pdf, const float2& _sample) const;
	CUDA_DEVICE CUDA_HOST float3 f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	template<typename T> void LoadTextures(T callback)
	{
		m_diffuseReflectance.LoadTextures(callback);
		m_specularReflectance.LoadTextures(callback);
		m_exponent.LoadTextures(callback);
	}
	TYPE_FUNC(phong)
};

#define ward_TYPE 10
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
	e_KernelTexture<float3> m_diffuseReflectance;
	e_KernelTexture<float3> m_specularReflectance;
	e_KernelTexture<float> m_alphaU;
	e_KernelTexture<float> m_alphaV;
	ward()
		: BSDF(EGlossyReflection | EDiffuseReflection)
	{
	}
	ward(EModelVariant type, const e_KernelTexture<float3>& d, const e_KernelTexture<float3>& s, const e_KernelTexture<float>& u, const e_KernelTexture<float>& v)
		: m_modelVariant(type), m_diffuseReflectance(d), m_specularReflectance(s), m_alphaU(u), m_alphaV(v), BSDF(EGlossyReflection | EDiffuseReflection)
	{
		MapParameters mp(make_float3(0), make_float2(0), Frame(make_float3(0,1,0)));
		float dAvg = y(m_diffuseReflectance.Evaluate(mp)), sAvg = y(m_specularReflectance.Evaluate(mp));
		m_specularSamplingWeight = sAvg / (dAvg + sAvg);
	}
	CUDA_DEVICE CUDA_HOST float3 sample(BSDFSamplingRecord &bRec, float &_pdf, const float2 &_sample) const;
	CUDA_DEVICE CUDA_HOST float3 f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	template<typename T> void LoadTextures(T callback)
	{
		m_diffuseReflectance.LoadTextures(callback);
		m_specularReflectance.LoadTextures(callback);
		m_alphaU.LoadTextures(callback);
		m_alphaV.LoadTextures(callback);
	}
	TYPE_FUNC(ward)
};

#define hk_TYPE 11
struct hk : public BSDF
{
	e_KernelTexture<float3> m_sigmaS;
	e_KernelTexture<float3> m_sigmaA;
	e_PhaseFunction m_phase;
	float m_thickness;
	hk()
		: BSDF(EGlossyReflection | EGlossyTransmission | EDeltaTransmission)
	{
	}
	hk(const e_KernelTexture<float3>& ss, const e_KernelTexture<float3>& sa, e_PhaseFunction& phase, float thickness)
		: m_sigmaS(ss), m_sigmaA(sa), m_phase(phase), m_thickness(thickness), BSDF(EGlossyReflection | EGlossyTransmission | EDeltaTransmission)
	{
	}
	CUDA_DEVICE CUDA_HOST float3 sample(BSDFSamplingRecord &bRec, float &_pdf, const float2 &_sample) const;
	CUDA_DEVICE CUDA_HOST float3 f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	template<typename T> void LoadTextures(T callback)
	{
		m_sigmaS.LoadTextures(callback);
		m_sigmaA.LoadTextures(callback);
	}
	TYPE_FUNC(hk)
};