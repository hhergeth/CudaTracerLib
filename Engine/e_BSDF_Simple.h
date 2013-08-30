#pragma once

#define diffuse_TYPE 1
struct diffuse : public BSDF
{
	e_KernelTexture<float3>	m_reflectance;
	diffuse()
		: BSDF(EDiffuseReflection)
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
	CUDA_DEVICE CUDA_HOST float3 sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &_sample) const;
	CUDA_DEVICE CUDA_HOST float3 f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	template<typename T> void LoadTextures(T callback) const
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
	CUDA_FUNC_IN float3 reflect(const float3 &wi) const {
		return make_float3(-wi.x, -wi.y, wi.z);
	}	
	CUDA_DEVICE CUDA_HOST float3 sample(BSDFSamplingRecord &bRec, float &_pdf, const float2& _sample) const;
	CUDA_DEVICE CUDA_HOST float3 f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	template<typename T> void LoadTextures(T callback) const
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
	CUDA_DEVICE CUDA_HOST float3 sample(BSDFSamplingRecord &bRec, float &_pdf, const float2 &_sample) const;
	CUDA_DEVICE CUDA_HOST float3 f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	template<typename T> void LoadTextures(T callback) const
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
	CUDA_DEVICE CUDA_HOST float3 sample(BSDFSamplingRecord &bRec, float &_pdf, const float2 &_sample) const;
	CUDA_DEVICE CUDA_HOST float3 f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	template<typename T> void LoadTextures(T callback) const
	{
		m_sigmaS.LoadTextures(callback);
		m_sigmaA.LoadTextures(callback);
	}
	TYPE_FUNC(hk)
};