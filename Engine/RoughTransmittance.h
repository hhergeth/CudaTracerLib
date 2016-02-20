#pragma once

#include "MicrofacetDistribution.h"

//Implementation and interface copied from Mitsuba.

namespace CudaTracerLib {

class RoughTransmittance
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
	RoughTransmittance() = default;
	CTL_EXPORT RoughTransmittance(const std::string& name);
	CTL_EXPORT void Free();
	CTL_EXPORT CUDA_DEVICE CUDA_HOST float Evaluate(float cosTheta, float alpha = 0.0f, float eta = 0.0f) const;
	CTL_EXPORT CUDA_DEVICE CUDA_HOST float EvaluateDiffuse(float alpha = 0, float eta = 0) const;
};

class RoughTransmittanceManager
{
public:
	CTL_EXPORT void static StaticInitialize(const std::string& a_Path);
	CTL_EXPORT void static StaticDeinitialize();
	CTL_EXPORT CUDA_DEVICE CUDA_HOST static float Evaluate(MicrofacetDistribution::EType type, float cosTheta, float alpha = 0.0f, float eta = 0.0f);
	CTL_EXPORT CUDA_DEVICE CUDA_HOST static float EvaluateDiffuse(MicrofacetDistribution::EType type, float alpha = 0, float eta = 0);
};

}