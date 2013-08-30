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

/*
struct coating : public BSDF
{
	BSDFALL* m_nested;
	float m_specularSamplingWeight;
	float m_eta, m_invEta;
	e_KernelTexture<float3> m_sigmaA;
	e_KernelTexture<float3> m_specularReflectance;
	float m_thickness;
	CUDA_FUNC_IN float3 sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const;
	CUDA_FUNC_IN float3 f(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	CUDA_FUNC_IN float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const;
	template<typename T> void LoadTextures(T callback) const
	{
		m_sigmaA.LoadTextures(callback);
		m_specularReflectance.LoadTextures(callback);
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

struct roughcoating : public BSDF
{
private:
	BSDFALL* bsdf;
public:
	CUDA_FUNC_IN float pdf(float3& a, float3& b);
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