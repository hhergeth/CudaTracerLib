#pragma once

#include "..\Base\FileStream.h"
#include "e_MicrofacetDistribution.h"

class e_RoughTransmittance
{
	size_t m_etaSamples;
	size_t m_alphaSamples;
	size_t m_thetaSamples;
	float m_etaMin, m_etaMax;
	float m_alphaMin, m_alphaMax;
	size_t m_transSize;
	size_t m_diffTransSize;
	float *m_transDevice, *m_diffTransDevice;
	float *m_transHost, *m_diffTransHost;
public:
	e_RoughTransmittance(){}
	e_RoughTransmittance(const std::string& name);
	void Free()
	{
		delete [] m_transHost;
		delete [] m_diffTransHost;
		CUDA_FREE(m_transDevice);
		CUDA_FREE(m_diffTransDevice);
	}
	CUDA_DEVICE CUDA_HOST float Evaluate(float cosTheta, float alpha = 0.0f, float eta = 0.0f) const;
	CUDA_DEVICE CUDA_HOST float EvaluateDiffuse(float alpha = 0, float eta = 0) const;
};

class e_RoughTransmittanceManager
{
public:
	void static StaticInitialize(const std::string& a_Path);
	void static StaticDeinitialize();
	CUDA_DEVICE CUDA_HOST static float Evaluate(MicrofacetDistribution::EType type, float cosTheta, float alpha = 0.0f, float eta = 0.0f);
	CUDA_DEVICE CUDA_HOST static float EvaluateDiffuse(MicrofacetDistribution::EType type, float alpha = 0, float eta = 0);
};