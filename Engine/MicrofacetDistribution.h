#pragma once

#include <Math/Vector.h>
#include <Math/Frame.h>

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
	};
	EType m_type;
	float m_alphaU, m_alphaV;
	bool m_sampleVisible;
	float m_exponentU, m_exponentV;
public:
	CUDA_FUNC_IN MicrofacetDistribution(EType type, float alpha, bool sampleVisible)
		: m_type(type), m_alphaU(alpha), m_alphaV(alpha), m_sampleVisible(sampleVisible), m_exponentU(0.0f), m_exponentV(0.0f)
	{
		m_alphaU = DMAX2(m_alphaU, 1e-4f);
		m_alphaV = DMAX2(m_alphaV, 1e-4f);
		if (m_type == EPhong)
			computePhongExponent();
	}

	CUDA_FUNC_IN MicrofacetDistribution(EType type, float alphaU, float alphaV, bool sampleVisible)
		: m_type(type), m_alphaU(alphaU), m_alphaV(alphaV), m_sampleVisible(sampleVisible),	m_exponentU(0.0f), m_exponentV(0.0f)
	{
		m_alphaU = DMAX2(m_alphaU, 1e-4f);
		m_alphaV = DMAX2(m_alphaV, 1e-4f);
		if (m_type == EPhong)
			computePhongExponent();
	}

	static bool getSampleVisible(EType type, bool sampleVisible)
	{
		return type == EType::EPhong ? false : sampleVisible;
	}

	CUDA_FUNC_IN EType getType() const { return m_type; }
	CUDA_FUNC_IN float getAlpha() const { return m_alphaU; }
	CUDA_FUNC_IN float getAlphaU() const { return m_alphaU; }
	CUDA_FUNC_IN float getAlphaV() const { return m_alphaV; }
	CUDA_FUNC_IN float getExponent() const { return m_exponentU; }
	CUDA_FUNC_IN float getExponentU() const { return m_exponentU; }
	CUDA_FUNC_IN float getExponentV() const { return m_exponentV; }
	CUDA_FUNC_IN bool getSampleVisible() const { return m_sampleVisible; }
	CUDA_FUNC_IN bool isAnisotropic() const { return m_alphaU != m_alphaV; }
	CUDA_FUNC_IN bool isIsotropic() const { return m_alphaU == m_alphaV; }

	CUDA_FUNC_IN void scaleAlpha(float value)
	{
		m_alphaU *= value;
		m_alphaV *= value;
		if (m_type == EPhong)
			computePhongExponent();
	}

	CTL_EXPORT CUDA_DEVICE CUDA_HOST float eval(const NormalizedT<Vec3f>& m) const;

	CUDA_FUNC_IN NormalizedT<Vec3f> sample(const NormalizedT<Vec3f>& wi, const Vec2f& sample, float& pdf) const
	{
		NormalizedT<Vec3f> m;
		if (m_sampleVisible) {
			m = sampleVisible(wi, sample);
			pdf = pdfVisible(wi, m);
		}
		else {
			m = sampleAll(sample, pdf);
		}
		return m;
	}

	CUDA_FUNC_IN NormalizedT<Vec3f> sample(const NormalizedT<Vec3f> &wi, const Vec2f& sample) const
	{
		NormalizedT<Vec3f> m;
		if (m_sampleVisible) {
			m = sampleVisible(wi, sample);
		}
		else {
			float pdf;
			m = sampleAll(sample, pdf);
		}
		return m;
	}

	CUDA_FUNC_IN float pdf(const NormalizedT<Vec3f>& wi, const NormalizedT<Vec3f>& m) const
	{
		if (m_sampleVisible)
			return pdfVisible(wi, m);
		else
			return pdfAll(m);
	}

	CTL_EXPORT CUDA_DEVICE CUDA_HOST NormalizedT<Vec3f> sampleAll(const Vec2f& sample, float& pdf) const;

	CUDA_FUNC_IN float pdfAll(const NormalizedT<Vec3f>& m) const
	{
		return eval(m) * Frame::cosTheta(m);
	}

	CTL_EXPORT CUDA_DEVICE CUDA_HOST NormalizedT<Vec3f> sampleVisible(const NormalizedT<Vec3f>& _wi, const Vec2f& sample) const;

	CUDA_FUNC_IN float pdfVisible(const NormalizedT<Vec3f>& wi, const NormalizedT<Vec3f>& m) const
	{
		if (Frame::cosTheta(wi) == 0)
			return 0.0f;
		return smithG1(wi, m) * absdot(wi, m) * eval(m) / std::abs(Frame::cosTheta(wi));
	}

	CTL_EXPORT CUDA_DEVICE CUDA_HOST float smithG1(const NormalizedT<Vec3f> &v, const NormalizedT<Vec3f> &m) const;

	CUDA_FUNC_IN float G(const NormalizedT<Vec3f> &wi, const NormalizedT<Vec3f> &wo, const NormalizedT<Vec3f> &m) const
	{
		return smithG1(wi, m) * smithG1(wo, m);
	}
private:
	CUDA_FUNC_IN float projectRoughness(const NormalizedT<Vec3f>& v) const {
		float invSinTheta2 = 1 / Frame::sinTheta2(v);

		if (isIsotropic() || invSinTheta2 <= 0)
			return m_alphaU;

		float cosPhi2 = v.x * v.x * invSinTheta2;
		float sinPhi2 = v.y * v.y * invSinTheta2;

		return std::sqrt(cosPhi2 * m_alphaU * m_alphaU + sinPhi2 * m_alphaV * m_alphaV);
	}

	CUDA_FUNC_IN float interpolatePhongExponent(const NormalizedT<Vec3f>& v) const {
		const float sinTheta2 = Frame::sinTheta2(v);

		if (isIsotropic() || sinTheta2 <= RCPOVERFLOW)
			return m_exponentU;

		float invSinTheta2 = 1 / sinTheta2;
		float cosPhi2 = v.x * v.x * invSinTheta2;
		float sinPhi2 = v.y * v.y * invSinTheta2;

		return m_exponentU * cosPhi2 + m_exponentV * sinPhi2;
	}

	CTL_EXPORT CUDA_DEVICE CUDA_HOST Vec2f sampleVisible11(float thetaI, Vec2f sample) const;

	CUDA_FUNC_IN void computePhongExponent()
	{
		m_exponentU = DMAX2(2.0f / (m_alphaU * m_alphaU) - 2.0f, 0.0f);
		m_exponentV = DMAX2(2.0f / (m_alphaV * m_alphaV) - 2.0f, 0.0f);
	}

	CUDA_FUNC_IN void sampleFirstQuadrant(float u1, float &phi, float &exponent) const
	{
		float cosPhi, sinPhi;
		phi = math::atan(
			math::sqrt((m_exponentU + 2.0f) / (m_exponentV + 2.0f)) *
			math::tan(PI * u1 * 0.5f));
		sincos(phi, &sinPhi, &cosPhi);
		/* Return the interpolated roughness */
		exponent = m_exponentU * cosPhi * cosPhi + m_exponentV * sinPhi * sinPhi;
	}
};

}