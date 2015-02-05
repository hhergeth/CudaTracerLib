#pragma once

#include "..\MathTypes.h"

struct MicrofacetDistribution
{
	enum EType {
		/// Beckmann distribution derived from Gaussian random surfaces
		EBeckmann         = 0,
		/// Long-tailed distribution proposed by Walter et al.
		EGGX              = 1,
		/// Classical Phong distribution
		EPhong            = 2,
		/// Anisotropic distribution by Ashikhmin and Shirley
		EAshikhminShirley = 3
	};
	EType m_type;
	CUDA_FUNC_IN float transformRoughness(float value) const {
		value = max(value, (float) 1e-3f);
		if (m_type == EPhong || m_type == EAshikhminShirley)
			value = max(2 / (value * value) - 2, (float) 0.1f);
		return value;
	}
	CUDA_FUNC_IN float eval(const Vec3f &m, float alpha) const {
		return eval(m, alpha, alpha);
	}
	CUDA_DEVICE CUDA_HOST float eval(const Vec3f &m, float alphaU, float alphaV) const;
	CUDA_FUNC_IN float pdf(const Vec3f &m, float alpha) const {
		return pdf(m, alpha, alpha);
	}
	CUDA_DEVICE CUDA_HOST float pdf(const Vec3f &m, float alphaU, float alphaV) const;
	CUDA_FUNC_IN void sampleFirstQuadrant(float alphaU, float alphaV, float u1, float u2,
			float &phi, float &cosTheta) const {
		if (alphaU == alphaV)
			phi = PI * u1 * 0.5f;
		else
			phi = std::atan(
				math::sqrt((alphaU + 1.0f) / (alphaV + 1.0f)) *
				std::tan(PI * u1 * 0.5f));
		const float cosPhi = std::cos(phi), sinPhi = std::sin(phi);
		cosTheta = std::pow(u2, 1.0f /
			(alphaU * cosPhi * cosPhi + alphaV * sinPhi * sinPhi + 1.0f));
	}
	CUDA_FUNC_IN Vec3f sample(const Vec2f &sample, float alpha) const {
		return MicrofacetDistribution::sample(sample, alpha, alpha);
	}
	CUDA_DEVICE CUDA_HOST Vec3f sample(const Vec2f &sample, float alphaU, float alphaV) const;
	CUDA_DEVICE CUDA_HOST Vec3f sample(const Vec2f &sample, float alphaU, float alphaV, float &pdf) const;
	CUDA_DEVICE CUDA_HOST float smithG1(const Vec3f &v, const Vec3f &m, float alpha) const;
	CUDA_FUNC_IN float G(const Vec3f &wi, const Vec3f &wo, const Vec3f &m, float alpha) const {
		return G(wi, wo, m, alpha, alpha);
	}
	CUDA_DEVICE CUDA_HOST float G(const Vec3f &wi, const Vec3f &wo, const Vec3f &m, float alphaU, float alphaV) const;
};