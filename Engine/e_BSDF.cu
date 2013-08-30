#include "e_BSDF.h"

float MicrofacetDistribution::eval(const float3 &m, float alphaU, float alphaV) const
{
	if (Frame::cosTheta(m) <= 0)
		return 0.0f;

	float result;
	switch (m_type) {
		case EBeckmann: {
				/* Beckmann distribution function for Gaussian random surfaces */
				const float exponent  = Frame::tanTheta2(m) / (alphaU*alphaU);
				const float cosTheta2 = Frame::cosTheta2(m);

				result = exp(-exponent) /
					(PI * alphaU*alphaU*cosTheta2*cosTheta2);
			}
			break;

		case EGGX: {
				/* Empirical GGX distribution function for rough surfaces */
				const float tanTheta2 = Frame::tanTheta2(m),
						    cosTheta2 = Frame::cosTheta2(m);

				const float root = alphaU / (cosTheta2 *
							(alphaU*alphaU + tanTheta2));

				result = INV_PI * (root * root);
			}
			break;

		case EPhong: {
				/* Phong distribution function */
				result = (alphaU + 2) * INV_TWOPI
						* std::pow(Frame::cosTheta(m), alphaU);
			}
			break;

		case EAshikhminShirley: {
				const float cosTheta = Frame::cosTheta(m);
				const float ds = 1 - cosTheta * cosTheta;
				if (ds < 0)
					return 0.0f;
				const float exponent = (alphaU * m.x * m.x
						+ alphaV * m.y * m.y) / ds;
				result = sqrtf((alphaU + 2) * (alphaV + 2))
					* INV_TWOPI * std::pow(cosTheta, exponent);
			}
			break;
	}

	/* Prevent potential numerical issues in other stages of the model */
	if (result < 1e-20f)
		result = 0;

	return result;
}

float MicrofacetDistribution::pdf(const float3 &m, float alphaU, float alphaV) const
{
	/* Usually, this is just D(m) * cos(theta_M) */
	if (m_type != EAshikhminShirley)
		return eval(m, alphaU, alphaV) * Frame::cosTheta(m);

	/* For the Ashikhmin-Shirley model, the sampling density
		does not include the cos(theta_M) factor, and the
		normalization is slightly different than in eval(). */
	const float cosTheta = Frame::cosTheta(m);
	const float ds = 1 - cosTheta * cosTheta;
	if (ds < 0)
		return 0.0f;
	const float exponent = (alphaU * m.x * m.x
			+ alphaV * m.y * m.y) / ds;
	float result = sqrtf((alphaU + 1) * (alphaV + 1))
		* INV_TWOPI * std::pow(cosTheta, exponent);

	/* Prevent potential numerical issues in other stages of the model */
	if (result < 1e-20f)
		result = 0;

	return result;
}

float3 MicrofacetDistribution::sample(const float2 &sample, float alphaU, float alphaV) const
{
	/* The azimuthal component is always selected
		uniformly regardless of the distribution */
	float cosThetaM = 0.0f, phiM = (2.0f * PI) * sample.y;

	switch (m_type) {
		case EBeckmann: {
				float tanThetaMSqr = -alphaU*alphaU * log(1.0f - sample.x);
				cosThetaM = 1.0f / sqrtf(1 + tanThetaMSqr);
			}
			break;

		case EGGX: {
				float tanThetaMSqr = alphaU * alphaU * sample.x / (1.0f - sample.x);
				cosThetaM = 1.0f / sqrtf(1 + tanThetaMSqr);
			}
			break;

		case EPhong: {
				cosThetaM = std::pow(sample.x, 1 / (alphaU + 2));
			}
			break;

		case EAshikhminShirley: {
				/* Sampling method based on code from PBRT */
				if (sample.x < 0.25f) {
					sampleFirstQuadrant(alphaU, alphaV,
						4 * sample.x, sample.y, phiM, cosThetaM);
				} else if (sample.x < 0.5f) {
					sampleFirstQuadrant(alphaU, alphaV,
						4 * (0.5f - sample.x), sample.y, phiM, cosThetaM);
					phiM = PI - phiM;
				} else if (sample.x < 0.75f) {
					sampleFirstQuadrant(alphaU, alphaV,
						4 * (sample.x - 0.5f), sample.y, phiM, cosThetaM);
					phiM += PI;
				} else {
					sampleFirstQuadrant(alphaU, alphaV,
						4 * (1 - sample.x), sample.y, phiM, cosThetaM);
					phiM = 2 * PI - phiM;
				}
			}
			break;
	}

	const float sinThetaM = sqrtf(
		MAX((float) 0, 1 - cosThetaM*cosThetaM));

	float sinPhiM, cosPhiM;
	sincos(phiM, &sinPhiM, &cosPhiM);

	return make_float3(
		sinThetaM * cosPhiM,
		sinThetaM * sinPhiM,
		cosThetaM
	);
}

float3 MicrofacetDistribution::sample(const float2 &sample, float alphaU, float alphaV, float &pdf) const
{
	/* The azimuthal component is always selected
		uniformly regardless of the distribution */
	float cosThetaM = 0.0f;

	switch (m_type) {
		case EBeckmann: {
				float tanThetaMSqr = -alphaU*alphaU * logf(1.0f - sample.x);
				cosThetaM = 1.0f / sqrtf(1 + tanThetaMSqr);
				float cosThetaM2 = cosThetaM * cosThetaM,
						cosThetaM3 = cosThetaM2 * cosThetaM;
				pdf = (1.0f - sample.x) / (PI * alphaU*alphaU * cosThetaM3);
			}
			break;

		case EGGX: {
				float alphaUSqr = alphaU * alphaU;
				float tanThetaMSqr = alphaUSqr * sample.x / (1.0f - sample.x);
				cosThetaM = 1.0f / sqrtf(1 + tanThetaMSqr);

				float cosThetaM2 = cosThetaM * cosThetaM,
						cosThetaM3 = cosThetaM2 * cosThetaM,
						temp = alphaUSqr + tanThetaMSqr;

				pdf = INV_PI * alphaUSqr / (cosThetaM3 * temp * temp);
			}
			break;

		case EPhong: {
				float exponent = 1 / (alphaU + 2);
				cosThetaM = powf(sample.x, exponent);
				pdf = (alphaU + 2) * INV_TWOPI * std::powf(sample.x, (alphaU+1) * exponent);
			}
			break;

		case EAshikhminShirley: {
				float phiM;

				/* Sampling method based on code from PBRT */
				if (sample.x < 0.25f) {
					sampleFirstQuadrant(alphaU, alphaV,
						4 * sample.x, sample.y, phiM, cosThetaM);
				} else if (sample.x < 0.5f) {
					sampleFirstQuadrant(alphaU, alphaV,
						4 * (0.5f - sample.x), sample.y, phiM, cosThetaM);
					phiM = PI - phiM;
				} else if (sample.x < 0.75f) {
					sampleFirstQuadrant(alphaU, alphaV,
						4 * (sample.x - 0.5f), sample.y, phiM, cosThetaM);
					phiM += PI;
				} else {
					sampleFirstQuadrant(alphaU, alphaV,
						4 * (1 - sample.x), sample.y, phiM, cosThetaM);
					phiM = 2 * PI - phiM;
				}
				const float sinThetaM = sqrtf(
						MAX((float) 0, 1 - cosThetaM*cosThetaM)),
						sinPhiM = std::sin(phiM),
						cosPhiM = std::cos(phiM);

				const float exponent = alphaU * cosPhiM*cosPhiM
						+ alphaV * sinPhiM*sinPhiM;
				pdf = sqrtf((alphaU + 1) * (alphaV + 1))
					* INV_TWOPI * std::pow(cosThetaM, exponent);

				/* Prevent potential numerical issues in other stages of the model */
				if (pdf < 1e-20f)
					pdf = 0;

				return make_float3(
					sinThetaM * cosPhiM,
					sinThetaM * sinPhiM,
					cosThetaM
				);
			}
			break;
	}

	/* Prevent potential numerical issues in other stages of the model */
	if (pdf < 1e-20f)
		pdf = 0;

	const float sinThetaM = sqrtf(
		MAX((float) 0, 1 - cosThetaM*cosThetaM));
	float phiM = (2.0f * PI) * sample.y;
	return make_float3(
		sinThetaM * std::cos(phiM),
		sinThetaM * std::sin(phiM),
		cosThetaM
	);
}

float MicrofacetDistribution::G(const float3 &wi, const float3 &wo, const float3 &m, float alphaU, float alphaV) const
{
	if (m_type != EAshikhminShirley) {
		return smithG1(wi, m, alphaU)
				* smithG1(wo, m, alphaU);
	} else {
		/* Can't see the back side from the front and vice versa */
		if (dot(wi, m) * Frame::cosTheta(wi) <= 0 ||
			dot(wo, m) * Frame::cosTheta(wo) <= 0)
			return 0.0f;

		/* Infinite groove shadowing/masking */
		const float nDotM  = Frame::cosTheta(m),
					nDotWo = Frame::cosTheta(wo),
					nDotWi = Frame::cosTheta(wi),
					woDotM = dot(wo, m),
					wiDotM = dot(wi, m);

		return MIN((float) 1,
			std::min(abs(2 * nDotM * nDotWo / woDotM),
						abs(2 * nDotM * nDotWi / wiDotM)));
	}
}

float MicrofacetDistribution::smithG1(const float3 &v, const float3 &m, float alpha) const
{
	const float tanTheta = abs(Frame::tanTheta(v));

	/* perpendicular incidence -- no shadowing/masking */
	if (tanTheta == 0.0f)
		return 1.0f;

	/* Can't see the back side from the front and vice versa */
	if (dot(v, m) * Frame::cosTheta(v) <= 0)
		return 0.0f;

	switch (m_type) {
		case EPhong:
		case EBeckmann: {
				float a;
				/* Approximation recommended by Bruce Walter: Use
					the Beckmann shadowing-masking function with
					specially chosen roughness value */
				if (m_type != EBeckmann)
					a = sqrtf(0.5f * alpha + 1) / tanTheta;
				else
					a = 1.0f / (alpha * tanTheta);

				if (a >= 1.6f)
					return 1.0f;

				/* Use a fast and accurate (<0.35% rel. error) rational
					approximation to the shadowing-masking function */
				const float aSqr = a * a;
				return (3.535f * a + 2.181f * aSqr)
						/ (1.0f + 2.276f * a + 2.577f * aSqr);
			}
			break;

		case EGGX: {
				const float root = alpha * tanTheta;
				return 2.0f / (1.0f + sqrtf(1.0f + root*root));
			}
			break;
	}
}