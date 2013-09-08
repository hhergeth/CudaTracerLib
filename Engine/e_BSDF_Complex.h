#pragma once

#include "e_BSDF_Simple.h"

struct BSDFALL;

/*
	CUDA_FUNC_IN float3 sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const
	{

	}
	CUDA_FUNC_IN float3 f(const BSDFSamplingRecord &bRec, EMeasure measure) const
	{

	}
	CUDA_FUNC_IN float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const
	{
		
	}
	template<typename T> void LoadTextures(T callback) const
	{

	}
*/

#define coating_TYPE 13
struct coating : public BSDF
{
	BSDFFirst m_nested;
	float m_specularSamplingWeight;
	float m_eta, m_invEta;
	e_KernelTexture<Spectrum> m_sigmaA;
	e_KernelTexture<Spectrum> m_specularReflectance;
	float m_thickness;
	coating()
		: BSDF(EDeltaReflection)
	{
	}
	coating(BSDFFirst& nested, float eta, float thickness, e_KernelTexture<Spectrum>& sig)
		: BSDF(EDeltaReflection | nested.getType()), m_nested(nested), m_eta(eta), m_invEta(1.0f / eta), m_thickness(thickness), m_sigmaA(sig)
	{
		m_specularReflectance = CreateTexture(0, Spectrum(1.0f));
		MapParameters mp(make_float3(0), make_float2(0), Frame(make_float3(0,1,0)));
		float avgAbsorption = (m_sigmaA.Evaluate(mp)*(-2*m_thickness)).exp().average();
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
	template<typename T> static coating Create(const T& val, float eta, float thickness, e_KernelTexture<Spectrum>& sig)
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
/*
struct roughcoating : public BSDF
{
	BSDFFirst m_nested;
	MicrofacetDistribution m_distribution;


	CUDA_DEVICE CUDA_HOST Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const;
	CUDA_DEVICE CUDA_HOST Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_DEVICE CUDA_HOST float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	template<typename T> void LoadTextures(T callback)
	{
		m_sigmaA.LoadTextures(callback);
		m_specularReflectance.LoadTextures(callback);
		m_nested.LoadTextures(callback);
	}
};

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