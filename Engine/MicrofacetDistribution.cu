#include "MicrofacetDistribution.h"
#include <Math/Frame.h>

namespace CudaTracerLib {

float MicrofacetDistribution::eval(const NormalizedT<Vec3f>& m) const
{
	if (Frame::cosTheta(m) <= 0)
		return 0.0f;

	float cosTheta2 = Frame::cosTheta2(m);
	float beckmannExponent = ((m.x*m.x) / (m_alphaU * m_alphaU)	+ (m.y*m.y) / (m_alphaV * m_alphaV)) / cosTheta2;

	float result;
	switch (m_type) {
		case EBeckmann: {
			/* Beckmann distribution function for Gaussian random surfaces */
			result = math::exp(-beckmannExponent) /	(PI * m_alphaU * m_alphaV * cosTheta2 * cosTheta2);
		}
		break;

		case EGGX: {
			/* Empirical GGX distribution function for rough surfaces */
			float root = (1 + beckmannExponent) * cosTheta2;
			result = 1.0f / (PI * m_alphaU * m_alphaV * root * root);
		}
		break;

		case EPhong: {
			/* Phong distribution function */
			float exponent = interpolatePhongExponent(m);
			result = math::sqrt((m_exponentU + 2) * (m_exponentV + 2)) * INV_TWOPI * math::pow(Frame::cosTheta(m), exponent);
		}
		break;
	}

	/* Prevent potential numerical issues in other stages of the model */
	if (result < 1e-20f)
		result = 0;

	return result;
}

NormalizedT<Vec3f> MicrofacetDistribution::sampleAll(const Vec2f& sample, float& pdf) const
{
	/* The azimuthal component is always selected
	uniformly regardless of the distribution */
	float cosThetaM = 0.0f;
	float sinPhiM, cosPhiM;
	float alphaSqr;

	switch (m_type) {
		case EBeckmann: {
			/* Beckmann distribution function for Gaussian random surfaces */
			if (isIsotropic()) {
				/* Sample phi component (isotropic case) */
				sincos((2.0f * PI) * sample.y, &sinPhiM, &cosPhiM);

				alphaSqr = m_alphaU * m_alphaU;
			}
			else {
				/* Sample phi component (anisotropic case) */
				float phiM = math::atan(m_alphaV / m_alphaU *
					math::tan(PI + 2 * PI*sample.y)) + PI * math::floor(2 * sample.y + 0.5f);
				sincos(phiM, &sinPhiM, &cosPhiM);

				float cosSc = cosPhiM / m_alphaU, sinSc = sinPhiM / m_alphaV;
				alphaSqr = 1.0f / (cosSc*cosSc + sinSc*sinSc);
			}

			/* Sample theta component */
			float tanThetaMSqr = alphaSqr * -math::log(1.0f - sample.x);
			cosThetaM = 1.0f / math::sqrt(1.0f + tanThetaMSqr);

			/* Compute probability density of the sampled position */
			pdf = (1.0f - sample.x) / (PI*m_alphaU*m_alphaV*cosThetaM*cosThetaM*cosThetaM);
		}
		break;

		case EGGX: {
			/* GGX / Trowbridge-Reitz distribution function for rough surfaces */
			if (isIsotropic()) {
				/* Sample phi component (isotropic case) */
				sincos((2.0f * PI) * sample.y, &sinPhiM, &cosPhiM);

				/* Sample theta component */
				alphaSqr = m_alphaU*m_alphaU;
			}
			else {
				/* Sample phi component (anisotropic case) */
				float phiM = math::atan(m_alphaV / m_alphaU *
					math::tan(PI + 2 * PI*sample.y)) + PI * math::floor(2 * sample.y + 0.5f);
				sincos(phiM, &sinPhiM, &cosPhiM);

				float cosSc = cosPhiM / m_alphaU, sinSc = sinPhiM / m_alphaV;
				alphaSqr = 1.0f / (cosSc*cosSc + sinSc*sinSc);
			}

			/* Sample theta component */
			float tanThetaMSqr = alphaSqr * sample.x / (1.0f - sample.x);
			cosThetaM = 1.0f / math::sqrt(1.0f + tanThetaMSqr);

			/* Compute probability density of the sampled position */
			float temp = 1 + tanThetaMSqr / alphaSqr;
			pdf = INV_PI / (m_alphaU*m_alphaV*cosThetaM*cosThetaM*cosThetaM*temp*temp);
		}
		break;

		case EPhong: {
			float phiM;
			float exponent;
			if (isIsotropic()) {
				phiM = (2.0f * PI) * sample.y;
				exponent = m_exponentU;
			}
			else {
				/* Sampling method based on code from PBRT */
				if (sample.y < 0.25f) {
					sampleFirstQuadrant(4 * sample.y, phiM, exponent);
				}
				else if (sample.y < 0.5f) {
					sampleFirstQuadrant(4 * (0.5f - sample.y), phiM, exponent);
					phiM = PI - phiM;
				}
				else if (sample.y < 0.75f) {
					sampleFirstQuadrant(4 * (sample.y - 0.5f), phiM, exponent);
					phiM += PI;
				}
				else {
					sampleFirstQuadrant(4 * (1 - sample.y), phiM, exponent);
					phiM = 2 * PI - phiM;
				}
			}
			sincos(phiM, &sinPhiM, &cosPhiM);
			cosThetaM = math::pow(sample.x, 1.0f / (exponent + 2.0f));
			pdf = math::sqrt((m_exponentU + 2.0f) * (m_exponentV + 2.0f))
				* INV_TWOPI * math::pow(cosThetaM, exponent + 1.0f);
		}
		break;
	}

	/* Prevent potential numerical issues in other stages of the model */
	if (pdf < 1e-20f)
		pdf = 0;

	float sinThetaM = math::sqrt(DMAX2(0.0f, 1 - cosThetaM*cosThetaM));

	return NormalizedT<Vec3f>(sinThetaM * cosPhiM, sinThetaM * sinPhiM, cosThetaM);
}

NormalizedT<Vec3f> MicrofacetDistribution::sampleVisible(const NormalizedT<Vec3f>& _wi, const Vec2f& sample) const
{
	/* Step 1: stretch wi */
	auto wi = normalize(Vec3f(m_alphaU * _wi.x, m_alphaV * _wi.y, _wi.z));

	/* Get polar coordinates */
	float theta = 0, phi = 0;
	if (wi.z < 0.99999f)
	{
		theta = math::acos(wi.z);
		phi = math::atan2(wi.y, wi.x);
	}
	float sinPhi, cosPhi;
	sincos(phi, &sinPhi, &cosPhi);

	/* Step 2: simulate P22_{wi}(slope.x, slope.y, 1, 1) */
	Vec2f slope = sampleVisible11(theta, sample);

	/* Step 3: rotate */
	slope = Vec2f(
		cosPhi * slope.x - sinPhi * slope.y,
		sinPhi * slope.x + cosPhi * slope.y);

	/* Step 4: unstretch */
	slope.x *= m_alphaU;
	slope.y *= m_alphaV;

	/* Step 5: compute normal */
	float normalization = 1.0f / math::sqrt(slope.x*slope.x
		+ slope.y*slope.y + (float) 1.0);

	return NormalizedT<Vec3f>(-slope.x * normalization,	-slope.y * normalization, normalization);
}

Vec2f MicrofacetDistribution::sampleVisible11(float thetaI, Vec2f sample) const
{
	const float SQRT_PI_INV = 1 / math::sqrt(PI);
	Vec2f slope;

	switch (m_type) {
		case EBeckmann: {
			/* Special case (normal incidence) */
			if (thetaI < 1e-4f) {
				float sinPhi, cosPhi;
				float r = math::sqrt(-math::log(1.0f - sample.x));
				sincos(2 * PI * sample.y, &sinPhi, &cosPhi);
				return Vec2f(r * cosPhi, r * sinPhi);
			}

			/* The original inversion routine from the paper contained
			discontinuities, which causes issues for QMC integration
			and techniques like Kelemen-style MLT. The following code
			performs a numerical inversion with better behavior */
			float tanThetaI = math::tan(thetaI);
			float cotThetaI = 1 / tanThetaI;

			/* Search interval -- everything is parameterized
			in the erf() domain */
			float a = -1, c = math::erf(cotThetaI);
			float sample_x = DMAX2(sample.x, 1e-6f);

			/* Start with a good initial guess */
			//Float b = (1-sample_x) * a + sample_x * c;

			/* We can do better (inverse of an approximation computed in Mathematica) */
			float fit = 1 + thetaI*(-0.876f + thetaI * (0.4265f - 0.0594f*thetaI));
			float b = c - (1 + c) * math::pow(1 - sample_x, fit);

			/* Normalization factor for the CDF */
			float normalization = 1 / (1 + c + SQRT_PI_INV*	tanThetaI*math::exp(-cotThetaI*cotThetaI));

			int it = 0;
			while (++it < 10) {
				/* Bisection criterion -- the oddly-looking
				boolean expression are intentional to check
				for NaNs at little additional cost */
				if (!(b >= a && b <= c))
					b = 0.5f * (a + c);

				/* Evaluate the CDF and its derivative
				(i.e. the density function) */
				float invErf = math::erfinv(b);
				float value = normalization*(1 + b + SQRT_PI_INV*
					tanThetaI*math::exp(-invErf*invErf)) - sample_x;
				float derivative = normalization * (1
					- invErf*tanThetaI);

				if (math::abs(value) < 1e-5f)
					break;

				/* Update bisection intervals */
				if (value > 0)
					c = b;
				else
					a = b;

				b -= value / derivative;
			}

			/* Now convert back into a slope value */
			slope.x = math::erfinv(b);

			/* Simulate Y component */
			slope.y = math::erfinv(2.0f*DMAX2(sample.y, 1e-6f) - 1.0f);
		};
		break;

		case EGGX: {
			/* Special case (normal incidence) */
			if (thetaI < 1e-4f) {
				float sinPhi, cosPhi;
				float r = math::safe_sqrt(sample.x / (1 - sample.x));
				sincos(2 * PI * sample.y, &sinPhi, &cosPhi);
				return Vec2f(r * cosPhi, r * sinPhi);
			}

			/* Precomputations */
			float tanThetaI = math::tan(thetaI);
			float a = 1 / tanThetaI;
			float G1 = 2.0f / (1.0f + math::safe_sqrt(1.0f + 1.0f / (a*a)));

			/* Simulate X component */
			float A = 2.0f * sample.x / G1 - 1.0f;
			if (math::abs(A) == 1)
				A -= math::signum(A)*1e-7f;
			float tmp = 1.0f / (A*A - 1.0f);
			float B = tanThetaI;
			float D = math::safe_sqrt(B*B*tmp*tmp - (A*A - B*B) * tmp);
			float slope_x_1 = B * tmp - D;
			float slope_x_2 = B * tmp + D;
			slope.x = (A < 0.0f || slope_x_2 > 1.0f / tanThetaI) ? slope_x_1 : slope_x_2;

			/* Simulate Y component */
			float S;
			if (sample.y > 0.5f) {
				S = 1.0f;
				sample.y = 2.0f * (sample.y - 0.5f);
			}
			else {
				S = -1.0f;
				sample.y = 2.0f * (0.5f - sample.y);
			}

			/* Improved fit */
			float z =
				(sample.y * (sample.y * (sample.y * (-0.365728915865723f) + 0.790235037209296f) -
				0.424965825137544f) + 0.000152998850436920f) /
					(sample.y * (sample.y * (sample.y * (sample.y * 0.169507819808272f - 0.397203533833404f) -
				0.232500544458471f) + 1.0f) - 0.539825872510702f);

			slope.y = S * z * math::sqrt(1.0f + slope.x*slope.x);
		};
		break;

	};
	return slope;
}

float MicrofacetDistribution::smithG1(const NormalizedT<Vec3f> &v, const NormalizedT<Vec3f> &m) const
{
	/* Can't see the back side from the front and vice versa */
	if (dot(v, m) * Frame::cosTheta(v) <= 0)
		return 0.0f;

	const float tanTheta = math::abs(Frame::tanTheta(v));

	/* perpendicular incidence -- no shadowing/masking */
	if (tanTheta == 0.0f)
		return 1.0f;

	float alpha = projectRoughness(v);
	switch (m_type) {
		case EPhong:
		case EBeckmann: {
			float a = 1.0f / (alpha * tanTheta);
			if (a >= 1.6f)
				return 1.0f;

			/* Use a fast and accurate (<0.35% rel. error) rational
			approximation to the shadowing-masking function */
			float aSqr = a*a;
			return (3.535f * a + 2.181f * aSqr)
				/ (1.0f + 2.276f * a + 2.577f * aSqr);
		}

		case EGGX: {
			const float root = alpha * tanTheta;
			return 2.0f / (1.0f + math::hypot2(1.0f, root));
		}
	}
	return 0;
}

}