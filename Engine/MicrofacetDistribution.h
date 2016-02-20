#pragma once

#include <Math/Vector.h>

//Implementation and interface copied from Mitsuba.

namespace CudaTracerLib {

struct Spectrum;

struct MicrofacetDistribution
{
	enum EType {
		/// Beckmann distribution derived from Gaussian random surfaces
		EBeckmann = 0,
		/// Long-tailed distribution proposed by Walter et al.
		EGGX = 1,
		/// Classical Phong distribution
		EPhong = 2,
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
	CUDA_FUNC_IN float eval(const NormalizedT<Vec3f> &m, float alpha) const {
		return eval(m, alpha, alpha);
	}
	CTL_EXPORT CUDA_DEVICE CUDA_HOST float eval(const NormalizedT<Vec3f> &m, float alphaU, float alphaV) const;
	CUDA_FUNC_IN float pdf(const NormalizedT<Vec3f> &m, float alpha) const {
		return pdf(m, alpha, alpha);
	}
	CTL_EXPORT CUDA_DEVICE CUDA_HOST float pdf(const NormalizedT<Vec3f> &m, float alphaU, float alphaV) const;
	CUDA_FUNC_IN void sampleFirstQuadrant(float alphaU, float alphaV, float u1, float u2,
		float &phi, float &cosTheta) const {
		if (alphaU == alphaV)
			phi = PI * u1 * 0.5f;
		else
			phi = math::atan(
			math::sqrt((alphaU + 1.0f) / (alphaV + 1.0f)) *
			math::tan(PI * u1 * 0.5f));
		const float cosPhi = math::cos(phi), sinPhi = math::sin(phi);
		cosTheta = math::pow(u2, 1.0f /
			(alphaU * cosPhi * cosPhi + alphaV * sinPhi * sinPhi + 1.0f));
	}
	CUDA_FUNC_IN NormalizedT<Vec3f> sample(const Vec2f &sample, float alpha) const {
		return MicrofacetDistribution::sample(sample, alpha, alpha);
	}
	CTL_EXPORT CUDA_DEVICE CUDA_HOST NormalizedT<Vec3f> sample(const Vec2f &sample, float alphaU, float alphaV) const;
	CTL_EXPORT CUDA_DEVICE CUDA_HOST NormalizedT<Vec3f> sample(const Vec2f &sample, float alphaU, float alphaV, float &pdf) const;
	CTL_EXPORT CUDA_DEVICE CUDA_HOST float smithG1(const NormalizedT<Vec3f> &v, const NormalizedT<Vec3f> &m, float alpha) const;
	CUDA_FUNC_IN float G(const NormalizedT<Vec3f> &wi, const NormalizedT<Vec3f> &wo, const NormalizedT<Vec3f> &m, float alpha) const {
		return G(wi, wo, m, alpha, alpha);
	}
	CTL_EXPORT CUDA_DEVICE CUDA_HOST float G(const NormalizedT<Vec3f> &wi, const NormalizedT<Vec3f> &wo, const NormalizedT<Vec3f> &m, float alphaU, float alphaV) const;
};

}