#include "e_BSDF.h"

Spectrum roughdiffuse::f(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	if (!(bRec.typeMask & EGlossyReflection) || measure != ESolidAngle
		|| Frame::cosTheta(bRec.wi) <= 0
		|| Frame::cosTheta(bRec.wo) <= 0)
		return Spectrum(0.0f);

	/* Conversion from Beckmann-style RMS roughness to
		Oren-Nayar-style slope-area variance. The factor
		of 1/sqrt(2) was found to be a perfect fit up
		to extreme roughness values (>.5), after which
		the match is not as good anymore */
	const float conversionFactor = 1 / sqrtf((float) 2);

	float sigma = m_alpha.Evaluate(bRec.map).average() * conversionFactor;

	const float sigma2 = sigma*sigma;

	float sinThetaI = Frame::sinTheta(bRec.wi),
			sinThetaO = Frame::sinTheta(bRec.wo);

	float cosPhiDiff = 0;
	if (sinThetaI > EPSILON && sinThetaO > EPSILON) {
		/* Compute cos(phiO-phiI) using the half-angle formulae */
		float sinPhiI = Frame::sinPhi(bRec.wi),
				cosPhiI = Frame::cosPhi(bRec.wi),
				sinPhiO = Frame::sinPhi(bRec.wo),
				cosPhiO = Frame::cosPhi(bRec.wo);
		cosPhiDiff = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
	}

	if (m_useFastApprox) {
		float A = 1.0f - 0.5f * sigma2 / (sigma2 + 0.33f),
				B = 0.45f * sigma2 / (sigma2 + 0.09f),
				sinAlpha, tanBeta;

		if (Frame::cosTheta(bRec.wi) > Frame::cosTheta(bRec.wo)) {
			sinAlpha = sinThetaO;
			tanBeta = sinThetaI / Frame::cosTheta(bRec.wi);
		} else {
			sinAlpha = sinThetaI;
			tanBeta = sinThetaO / Frame::cosTheta(bRec.wo);
		}

		return m_reflectance.Evaluate(bRec.map)
			* (INV_PI * Frame::cosTheta(bRec.wo) * (A + B
			* MAX(cosPhiDiff, (float) 0.0f) * sinAlpha * tanBeta));
	} else {
		float sinThetaI = Frame::sinTheta(bRec.wi),
				sinThetaO = Frame::sinTheta(bRec.wo),
				thetaI = math::safe_acos(Frame::cosTheta(bRec.wi)),
				thetaO = math::safe_acos(Frame::cosTheta(bRec.wo)),
				alpha = MAX(thetaI, thetaO),
				beta = MIN(thetaI, thetaO);

		float sinAlpha, sinBeta, tanBeta;
		if (Frame::cosTheta(bRec.wi) > Frame::cosTheta(bRec.wo)) {
			sinAlpha = sinThetaO; sinBeta = sinThetaI;
			tanBeta = sinThetaI / Frame::cosTheta(bRec.wi);
		} else {
			sinAlpha = sinThetaI; sinBeta = sinThetaO;
			tanBeta = sinThetaO / Frame::cosTheta(bRec.wo);
		}

		float tmp = sigma2 / (sigma2 + 0.09f),
				tmp2 = (4*INV_PI*INV_PI) * alpha * beta,
				tmp3 = 2*beta*INV_PI;

		float C1 = 1.0f - 0.5f * sigma2 / (sigma2 + 0.33f),
				C2 = 0.45f * tmp,
				C3 = 0.125f * tmp * tmp2 * tmp2,
				C4 = 0.17f * sigma2 / (sigma2 + 0.13f);

		if (cosPhiDiff > 0)
			C2 *= sinAlpha;
		else
			C2 *= sinAlpha - tmp3*tmp3*tmp3;

		/* Compute tan(0.5 * (alpha+beta)) using the half-angle formulae */
		float tanHalf = (sinAlpha + sinBeta) / (
				math::safe_sqrt(1.0f - sinAlpha * sinAlpha) +
				math::safe_sqrt(1.0f - sinBeta  * sinBeta));

		Spectrum rho = m_reflectance.Evaluate(bRec.map),
					snglScat = rho * (C1 + cosPhiDiff * C2 * tanBeta +
					(1.0f - abs(cosPhiDiff)) * C3 * tanHalf),
					dblScat = rho * rho * (C4 * (1.0f - cosPhiDiff*tmp3*tmp3));

		return  (snglScat + dblScat) * (INV_PI * Frame::cosTheta(bRec.wo));
	}
}

Spectrum dielectric::sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const
{
	bool sampleReflection   = (bRec.typeMask & EDeltaReflection);
	bool sampleTransmission = (bRec.typeMask & EDeltaTransmission);

	float cosThetaT;
	float F = MonteCarlo::fresnelDielectricExt(Frame::cosTheta(bRec.wi), cosThetaT, m_eta);

	if (sampleTransmission && sampleReflection) {
		float f0 = m_specularReflectance.Evaluate(bRec.map).average(), f1 = m_specularTransmittance.Evaluate(bRec.map).average(), f = F*f0/(f0+f1);
		if (sample.x <= f) {
			bRec.sampledType = EDeltaReflection;
			bRec.wo = reflect(bRec.wi);
			bRec.eta = 1.0f;
			pdf = F;
			
			return m_specularReflectance.Evaluate(bRec.map) / f * F;
		} else {
			bRec.sampledType = EDeltaTransmission;
			bRec.wo = refract(bRec.wi, cosThetaT);
			bRec.eta = cosThetaT < 0 ? m_eta : m_invEta;
			pdf = 1-f;

			float factor = (bRec.mode == ERadiance)
				? (cosThetaT < 0 ? m_invEta : m_eta) : 1.0f;

			return m_specularTransmittance.Evaluate(bRec.map) * (factor * factor) / (1 - f) * (1 - F);
		}
	} else if (sampleReflection) {
		bRec.sampledType = EDeltaReflection;
		bRec.wo = reflect(bRec.wi);
		bRec.eta = 1.0f;
		pdf = 1.0f;

		return m_specularReflectance.Evaluate(bRec.map);
	} else if (sampleTransmission) {
		bRec.sampledType = EDeltaTransmission;
		bRec.wo = refract(bRec.wi, cosThetaT);
		bRec.eta = cosThetaT < 0 ? m_eta : m_invEta;
		pdf = 1.0f;

		float factor = (bRec.mode == ERadiance)
			? (cosThetaT < 0 ? m_invEta : m_eta) : 1.0f;

		return m_specularTransmittance.Evaluate(bRec.map) * (factor * factor * (1-F));
	}

	return Spectrum(0.0f);
}

Spectrum dielectric::f(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	bool sampleReflection   = (bRec.typeMask & EDeltaReflection) && measure == EDiscrete;
	bool sampleTransmission = (bRec.typeMask & EDeltaTransmission) && measure == EDiscrete;

	float cosThetaT;
	float F = MonteCarlo::fresnelDielectricExt(Frame::cosTheta(bRec.wi), cosThetaT, m_eta);

	if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) >= 0) {
		if (!sampleReflection || abs(dot(reflect(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
			return Spectrum(0.0f);

		return m_specularReflectance.Evaluate(bRec.map) * F;
	} else {
		if (!sampleTransmission || abs(dot(refract(bRec.wi, cosThetaT), bRec.wo)-1) > DeltaEpsilon)
			return Spectrum(0.0f);

		/* Radiance must be scaled to account for the solid angle compression
			that occurs when crossing the interface. */
		float factor = (bRec.mode == ERadiance)
			? (cosThetaT < 0 ? m_invEta : m_eta) : 1.0f;

		return m_specularTransmittance.Evaluate(bRec.map)  * factor * factor * (1 - F);
	}
}

float dielectric::pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	bool sampleReflection   = (bRec.typeMask & EDeltaReflection) && measure == EDiscrete;
	bool sampleTransmission = (bRec.typeMask & EDeltaTransmission) && measure == EDiscrete;

	float cosThetaT;
	float F = MonteCarlo::fresnelDielectricExt(Frame::cosTheta(bRec.wi), cosThetaT, m_eta);

	if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) >= 0) {
		if (!sampleReflection || abs(dot(reflect(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
			return 0.0f;

		return sampleTransmission ? F : 1.0f;
	} else {
		if (!sampleTransmission || abs(dot(refract(bRec.wi, cosThetaT), bRec.wo)-1) > DeltaEpsilon)
			return 0.0f;

		return sampleReflection ? 1-F : 1.0f;
	}
}

float thindielectric::pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	bool sampleReflection   = (bRec.typeMask & EDeltaReflection) && measure == EDiscrete;
	bool sampleTransmission = (bRec.typeMask & ENull) && measure == EDiscrete;

	float R = MonteCarlo::fresnelDielectricExt(abs(Frame::cosTheta(bRec.wi)), m_eta), T = 1-R;

	// Account for internal reflections: R' = R + TRT + TR^3T + ..
	if (R < 1)
		R += T*T * R / (1-R*R);

	if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) >= 0) {
		if (!sampleReflection || abs(dot(reflect(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
			return 0.0f;

		return sampleTransmission ? R : 1.0f;
	} else {
		if (!sampleTransmission || abs(dot(transmit(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
			return 0.0f;

		return sampleReflection ? 1-R : 1.0f;
	}
}

Spectrum thindielectric::f(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	bool sampleReflection   = (bRec.typeMask & EDeltaReflection) && measure == EDiscrete;
	bool sampleTransmission = (bRec.typeMask & ENull) && measure == EDiscrete;

	float R = MonteCarlo::fresnelDielectricExt(abs(Frame::cosTheta(bRec.wi)), m_eta), T = 1-R;

	// Account for internal reflections: R' = R + TRT + TR^3T + ..
	if (R < 1)
		R += T*T * R / (1-R*R);

	if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) >= 0) {
		if (!sampleReflection || abs(dot(reflect(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
			return make_float3(0.0f);

		return m_specularReflectance.Evaluate(bRec.map) * R;
	} else {
		if (!sampleTransmission || abs(dot(transmit(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
			return make_float3(0.0f);

		return m_specularTransmittance.Evaluate(bRec.map) * (1 - R);
	}
}

Spectrum thindielectric::sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const
{
	bool sampleReflection   = (bRec.typeMask & EDeltaReflection);
	bool sampleTransmission = (bRec.typeMask & ENull);

	float R = MonteCarlo::fresnelDielectricExt(Frame::cosTheta(bRec.wi), m_eta), T = 1-R;

	// Account for internal reflections: R' = R + TRT + TR^3T + ..
	if (R < 1)
		R += T*T * R / (1-R*R);

	if (sampleTransmission && sampleReflection) {
		if (sample.x <= R) {
			bRec.sampledType = EDeltaReflection;
			bRec.wo = reflect(bRec.wi);
			bRec.eta = 1.0f;
			pdf = R;

			return m_specularReflectance.Evaluate(bRec.map);
		} else {
			bRec.sampledType = ENull;
			bRec.wo = transmit(bRec.wi);
			bRec.eta = 1.0f;
			pdf = 1-R;

			return m_specularTransmittance.Evaluate(bRec.map);
		}
	} else if (sampleReflection) {
		bRec.sampledType = EDeltaReflection;
		bRec.wo = reflect(bRec.wi);
		bRec.eta = 1.0f;
		pdf = 1.0f;

		return m_specularReflectance.Evaluate(bRec.map) * R;
	} else if (sampleTransmission) {
		bRec.sampledType = ENull;
		bRec.wo = transmit(bRec.wi);
		bRec.eta = 1.0f;
		pdf = 1.0f;

		return m_specularTransmittance.Evaluate(bRec.map) * (1-R);
	}

	return Spectrum(0.0f);
}

float roughdielectric::pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	if (measure != ESolidAngle)
		return 0.0f;

	/* Determine the type of interaction */
	bool hasReflection   = (bRec.typeMask & EGlossyReflection),
		 hasTransmission = (bRec.typeMask & EGlossyTransmission),
		 reflect         = Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) > 0;

	float3 H;
	float dwh_dwo;

	if (reflect) {
		/* Zero probability if this component was not requested */
		if (!(bRec.typeMask & EGlossyReflection))
			return 0.0f;

		/* Calculate the reflection half-vector */
		H = normalize(bRec.wo+bRec.wi);

		/* Jacobian of the half-direction mapping */
		dwh_dwo = 1.0f / (4.0f * dot(bRec.wo, H));
	} else {
		/* Zero probability if this component was not requested */
		if (!(bRec.typeMask & EGlossyTransmission))
			return 0.0f;

		/* Calculate the transmission half-vector */
		float eta = Frame::cosTheta(bRec.wi) > 0
			? m_eta : m_invEta;

		H = normalize(bRec.wi + bRec.wo*eta);

		/* Jacobian of the half-direction mapping */
		float sqrtDenom = dot(bRec.wi, H) + eta * dot(bRec.wo, H);
		dwh_dwo = (eta*eta * dot(bRec.wo, H)) / (sqrtDenom*sqrtDenom);
	}

	/* Ensure that the half-vector points into the
		same hemisphere as the macrosurface normal */
	H *= math::signum(Frame::cosTheta(H));

	/* Evaluate the roughness */
	float alphaU = m_distribution.transformRoughness(
				m_alphaU.Evaluate(bRec.map).average()),
			alphaV = m_distribution.transformRoughness(
				m_alphaV.Evaluate(bRec.map).average());

#if ENLARGE_LOBE_TRICK == 1
	Float factor = (1.2f - 0.2f * std::sqrt(
		abs(Frame::cosTheta(bRec.wi))));
	alphaU *= factor; alphaV *= factor;
#endif

	/* Evaluate the microsurface normal sampling density */
	float prob = m_distribution.pdf(H, alphaU, alphaV);

	if (hasTransmission && hasReflection) {
		float F = MonteCarlo::fresnelDielectricExt(dot(bRec.wi, H), m_eta);
		prob *= reflect ? F : (1-F);
	}

	return abs(prob * dwh_dwo);
}

Spectrum roughdielectric::f(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	if (measure != ESolidAngle)
		return Spectrum(0.0f);

	/* Determine the type of interaction */
	bool reflect = Frame::cosTheta(bRec.wi)
		* Frame::cosTheta(bRec.wo) > 0;

	float3 H;
	if (reflect) {
		/* Stop if this component was not requested */
		if (!(bRec.typeMask & EGlossyReflection))
			return Spectrum(0.0f);

		/* Calculate the reflection half-vector */
		H = normalize(bRec.wo+bRec.wi);
	} else {
		/* Stop if this component was not requested */
		if (!(bRec.typeMask & EGlossyTransmission))
			return Spectrum(0.0f);

		/* Calculate the transmission half-vector */
		float eta = Frame::cosTheta(bRec.wi) > 0
			? m_eta : m_invEta;

		H = normalize(bRec.wi + bRec.wo*eta);
	}

	/* Ensure that the half-vector points into the
		same hemisphere as the macrosurface normal */
	H *= math::signum(Frame::cosTheta(H));

	/* Evaluate the roughness */
	float alphaU = m_distribution.transformRoughness(
				m_alphaU.Evaluate(bRec.map).average()),
		  alphaV = m_distribution.transformRoughness(
				m_alphaV.Evaluate(bRec.map).average());

	/* Evaluate the microsurface normal distribution */
	const float D = m_distribution.eval(H, alphaU, alphaV);
	if (D == 0)
		return Spectrum(0.0f);

	/* Fresnel factor */
	const float F = MonteCarlo::fresnelDielectricExt(dot(bRec.wi, H), m_eta);

	/* Smith's shadow-masking function */
	const float G = m_distribution.G(bRec.wi, bRec.wo, H, alphaU, alphaV);

	if (reflect) {
		/* Calculate the total amount of reflection */
		float value = F * D * G /
			(4.0f * abs(Frame::cosTheta(bRec.wi)));

		return m_specularReflectance.Evaluate(bRec.map) * value;
	} else {
		float eta = Frame::cosTheta(bRec.wi) > 0.0f ? m_eta : m_invEta;

		/* Calculate the total amount of transmission */
		float sqrtDenom = dot(bRec.wi, H) + eta * dot(bRec.wo, H);
		float value = ((1 - F) * D * G * eta * eta
			* dot(bRec.wi, H) * dot(bRec.wo, H)) /
			(Frame::cosTheta(bRec.wi) * sqrtDenom * sqrtDenom);

		/* Missing term in the original paper: account for the solid angle
			compression when tracing radiance -- this is necessary for
			bidirectional methods */
		float factor = (bRec.mode == ERadiance)
			? (Frame::cosTheta(bRec.wi) > 0 ? m_invEta : m_eta) : 1.0f;

		return m_specularTransmittance.Evaluate(bRec.map)
			* abs(value * factor * factor);
	}
}

Spectrum roughdielectric::sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &_sample) const
{
	float2 sample = (_sample);

	bool hasReflection = (bRec.typeMask & EGlossyReflection),
		 hasTransmission = (bRec.typeMask & EGlossyTransmission),
		 sampleReflection = hasReflection;

	if (!hasReflection && !hasTransmission)
		return Spectrum(0.0f);

	/* Evaluate the roughness */
	float alphaU = m_distribution.transformRoughness(
				m_alphaU.Evaluate(bRec.map).average()),
		  alphaV = m_distribution.transformRoughness(
				m_alphaV.Evaluate(bRec.map).average());

#if ENLARGE_LOBE_TRICK == 1
	Float factor = (1.2f - 0.2f * std::sqrt(
		abs(Frame::cosTheta(bRec.wi))));
	Float sampleAlphaU = alphaU * factor,
			sampleAlphaV = alphaV * factor;
#else
	float sampleAlphaU = alphaU,
		  sampleAlphaV = alphaV;
#endif

	/* Sample M, the microsurface normal */
	float microfacetPDF;
	const float3 m = m_distribution.sample(sample,
			sampleAlphaU, sampleAlphaV, microfacetPDF);

	if (microfacetPDF == 0)
		return Spectrum(0.0f);

	pdf = microfacetPDF;

	float cosThetaT, numerator = 1.0f;
	float F = MonteCarlo::fresnelDielectricExt(dot(bRec.wi, m), cosThetaT, m_eta);

	if (hasReflection && hasTransmission) {
		if (bRec.rng->randomFloat() > F) {
			sampleReflection = false;
			pdf *= 1-F;
		} else {
			pdf *= F;
		}
	} else {
		numerator = hasReflection ? F : (1-F);
	}

	Spectrum result;
	float dwh_dwo;

	if (sampleReflection) {
		/* Perfect specular reflection based on the microsurface normal */
		bRec.wo = -VectorMath::reflect(bRec.wi, m);
		bRec.eta = 1.0f;
		bRec.sampledType = EGlossyReflection;

		/* Side check */
		if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) <= 0)
			return Spectrum(0.0f);

		result = m_specularReflectance.Evaluate(bRec.map);

		/* Jacobian of the half-direction mapping */
		dwh_dwo = 1.0f / (4.0f * dot(bRec.wo, m));
	} else {
		if (cosThetaT == 0)
			return Spectrum(0.0f);

		/* Perfect specular transmission based on the microsurface normal */
		bRec.wo = MonteCarlo::refract(bRec.wi, m, m_eta, cosThetaT);
		bRec.eta = cosThetaT < 0 ? m_eta : m_invEta;
		bRec.sampledType = EGlossyTransmission;

		/* Side check */
		if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) >= 0)
			return Spectrum(0.0f);

		/* Radiance must be scaled to account for the solid angle compression
			that occurs when crossing the interface. */
		float factor = (bRec.mode == ERadiance)
			? (cosThetaT < 0 ? m_invEta : m_eta) : (1.0f);

		result = m_specularTransmittance.Evaluate(bRec.map) * (factor * factor);

		/* Jacobian of the half-direction mapping */
		float sqrtDenom = dot(bRec.wi, m) + bRec.eta * dot(bRec.wo, m);
		dwh_dwo = (bRec.eta*bRec.eta * dot(bRec.wo, m)) / (sqrtDenom*sqrtDenom);
	}

	numerator *= m_distribution.eval(m, alphaU, alphaV)
		* m_distribution.G(bRec.wi, bRec.wo, m, alphaU, alphaV)
		* dot(bRec.wi, m);

	float denominator = microfacetPDF * Frame::cosTheta(bRec.wi);

	pdf *= abs(dwh_dwo);

	return result * abs(numerator / denominator);
}

Spectrum conductor::sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const
{
	bool sampleReflection   = (bRec.typeMask & EDeltaReflection);

	if (!sampleReflection || Frame::cosTheta(bRec.wi) <= 0)
		return Spectrum(0.0f);

	bRec.sampledType = EDeltaReflection;
	bRec.wo = reflect(bRec.wi);
	bRec.eta = 1.0f;
	pdf = 1;

	return m_specularReflectance.Evaluate(bRec.map) * MonteCarlo::fresnelConductorExact(Frame::cosTheta(bRec.wi), m_eta, m_k);
}

Spectrum conductor::f(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	bool sampleReflection   = (bRec.typeMask & EDeltaReflection);

	/* Verify that the provided direction pair matches an ideal
		specular reflection; tolerate some roundoff errors */
	if (!sampleReflection || measure != EDiscrete ||
		Frame::cosTheta(bRec.wi) <= 0 ||
		Frame::cosTheta(bRec.wo) <= 0 ||
		abs(dot(reflect(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
		return Spectrum(0.0f);

	return m_specularReflectance.Evaluate(bRec.map) * MonteCarlo::fresnelConductorExact(Frame::cosTheta(bRec.wi), m_eta, m_k);
}

float conductor::pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	bool sampleReflection   = (bRec.typeMask & EDeltaReflection);

	/* Verify that the provided direction pair matches an ideal
		specular reflection; tolerate some roundoff errors */
	if (!sampleReflection || measure != EDiscrete ||
		Frame::cosTheta(bRec.wi) <= 0 ||
		Frame::cosTheta(bRec.wo) <= 0 ||
		abs(dot(reflect(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
		return 0.0f;

	return 1.0f;
}

Spectrum roughconductor::sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const
{
	if (Frame::cosTheta(bRec.wi) < 0 ||	!(bRec.typeMask & EGlossyReflection))
		return Spectrum(0.0f);

	/* Evaluate the roughness */
	float alphaU = m_distribution.transformRoughness(
				m_alphaU.Evaluate(bRec.map).average()),
		  alphaV = m_distribution.transformRoughness(
				m_alphaV.Evaluate(bRec.map).average());

	/* Sample M, the microsurface normal */
	const float3 m = m_distribution.sample(sample,
		alphaU, alphaV, pdf);

	if (pdf == 0)
		return Spectrum(0.0f);

	/* Perfect specular reflection based on the microsurface normal */
	bRec.wo = -VectorMath::reflect(bRec.wi, m);
	bRec.eta = 1.0f;
	bRec.sampledType = EGlossyReflection;

	/* Side check */
	if (Frame::cosTheta(bRec.wo) <= 0)
		return Spectrum(0.0f);

	const Spectrum F = MonteCarlo::fresnelConductorExact(dot(bRec.wi, m),
			m_eta, m_k);

	float numerator = m_distribution.eval(m, alphaU, alphaV)
		* m_distribution.G(bRec.wi, bRec.wo, m, alphaU, alphaV)
		* dot(bRec.wi, m);

	float denominator = pdf * Frame::cosTheta(bRec.wi);

	/* Jacobian of the half-direction mapping */
	pdf /= 4.0f * dot(bRec.wo, m);

	return m_specularReflectance.Evaluate(bRec.map) * F	* (numerator / denominator);
}

Spectrum roughconductor::f(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	if (measure != ESolidAngle ||
		Frame::cosTheta(bRec.wi) < 0 ||
		Frame::cosTheta(bRec.wo) < 0 ||
		!(bRec.typeMask & EGlossyReflection))
		return Spectrum(0.0f);

	/* Calculate the reflection half-vector */
	float3 H = normalize(bRec.wo+bRec.wi);

	/* Evaluate the roughness */
	float alphaU = m_distribution.transformRoughness(m_alphaU.Evaluate(bRec.map).average()),
		  alphaV = m_distribution.transformRoughness(m_alphaV.Evaluate(bRec.map).average());

	/* Evaluate the microsurface normal distribution */
	const float D = m_distribution.eval(H, alphaU, alphaV);
	if (D == 0)
		return Spectrum(0.0f);

	/* Fresnel factor */
	const Spectrum F = MonteCarlo::fresnelConductorExact(dot(bRec.wi, H), m_eta, m_k);

	/* Smith's shadow-masking function */
	const float G = m_distribution.G(bRec.wi, bRec.wo, H, alphaU, alphaV);

	/* Calculate the total amount of reflection */
	float value = D * G / (4.0f * Frame::cosTheta(bRec.wi));

	return m_specularReflectance.Evaluate(bRec.map) * F * value;
}

float roughconductor::pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	if (measure != ESolidAngle ||
		Frame::cosTheta(bRec.wi) < 0 ||
		Frame::cosTheta(bRec.wo) < 0 ||
		!(bRec.typeMask & EGlossyReflection))
		return 0.0f;

	/* Calculate the reflection half-vector */
	float3 H = normalize(bRec.wo+bRec.wi);

	/* Evaluate the roughness */
	float alphaU = m_distribution.transformRoughness(m_alphaU.Evaluate(bRec.map).average()),
			alphaV = m_distribution.transformRoughness(m_alphaV.Evaluate(bRec.map).average());

	return m_distribution.pdf(H, alphaU, alphaV) / (4 * AbsDot(bRec.wo, H));
}

Spectrum plastic::sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const
{
	bool hasSpecular   = (bRec.typeMask & EDeltaReflection);
	bool hasDiffuse = (bRec.typeMask & EDiffuseReflection);

	if ((!hasDiffuse && !hasSpecular) || Frame::cosTheta(bRec.wi) <= 0)
		return Spectrum(0.0f);

	float Fi = MonteCarlo::fresnelDielectricExt(Frame::cosTheta(bRec.wi), m_eta);

	bRec.eta = 1.0f;
	if (hasDiffuse && hasSpecular) {
		float probSpecular = (Fi*m_specularSamplingWeight) /
			(Fi*m_specularSamplingWeight +
			(1-Fi) * (1-m_specularSamplingWeight));

		/* Importance sample wrt. the Fresnel reflectance */
		if (sample.x < probSpecular) {
			bRec.sampledType = EDeltaReflection;
			bRec.wo = reflect(bRec.wi);

			pdf = probSpecular;
			return m_specularReflectance.Evaluate(bRec.map)
				* Fi / probSpecular;
		} else {
			bRec.sampledType = EDiffuseReflection;
			bRec.wo = Warp::squareToCosineHemisphere(make_float2(
				(sample.x - probSpecular) / (1 - probSpecular),
				sample.y
			));
			float Fo = MonteCarlo::fresnelDielectricExt(Frame::cosTheta(bRec.wo), m_eta);

			Spectrum diff = m_diffuseReflectance.Evaluate(bRec.map);
			if (m_nonlinear)
				diff = diff / (Spectrum(1.0f) - diff*m_fdrInt);
			else
				diff = diff / (1 - m_fdrInt);

			pdf = (1-probSpecular) *
				Warp::squareToCosineHemispherePdf(bRec.wo);

			return diff * (m_invEta2 * (1-Fi) * (1-Fo) / (1-probSpecular));
		}
	} else if (hasSpecular) {
		bRec.sampledType = EDeltaReflection;
		bRec.wo = reflect(bRec.wi);
		pdf = 1;
		return m_specularReflectance.Evaluate(bRec.map) * Fi;
	} else {
		bRec.sampledType = EDiffuseReflection;
		bRec.wo = Warp::squareToCosineHemisphere(sample);
		float Fo = MonteCarlo::fresnelDielectricExt(Frame::cosTheta(bRec.wo), m_eta);

		Spectrum diff = m_diffuseReflectance.Evaluate(bRec.map);
		if (m_nonlinear)
			diff = diff / (Spectrum(1.0f) - diff*m_fdrInt);
		else
			diff = diff / (1 - m_fdrInt);

		pdf = Warp::squareToCosineHemispherePdf(bRec.wo);

		return diff * (m_invEta2 * (1-Fi) * (1-Fo));
	}
}

Spectrum plastic::f(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	bool hasSpecular   = (bRec.typeMask & EDeltaReflection)	&& measure == EDiscrete;
	bool hasDiffuse = (bRec.typeMask & EDiffuseReflection) && measure == ESolidAngle;

	if (Frame::cosTheta(bRec.wo) <= 0 || Frame::cosTheta(bRec.wi) <= 0)
		return Spectrum(0.0f);

	float Fi = MonteCarlo::fresnelDielectricExt(Frame::cosTheta(bRec.wi), m_eta);

	if (hasSpecular) {
		/* Check if the provided direction pair matches an ideal
			specular reflection; tolerate some roundoff errors */
		if (abs(dot(reflect(bRec.wi), bRec.wo)-1) < DeltaEpsilon)
			return m_specularReflectance.Evaluate(bRec.map) * Fi;
	} else if (hasDiffuse) {
		float Fo = MonteCarlo::fresnelDielectricExt(Frame::cosTheta(bRec.wo), m_eta);

		Spectrum diff = m_diffuseReflectance.Evaluate(bRec.map);

		if (m_nonlinear)
			diff = diff / (Spectrum(1.0f) - diff * m_fdrInt);
		else
			diff = diff / (1 - m_fdrInt);

		return diff * (Warp::squareToCosineHemispherePdf(bRec.wo)
			* m_invEta2 * (1-Fi) * (1-Fo));
	}

	return Spectrum(0.0f);
}

float plastic::pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	bool hasSpecular   = (bRec.typeMask & EDeltaReflection);
	bool hasDiffuse = (bRec.typeMask & EDiffuseReflection);

	if (Frame::cosTheta(bRec.wo) <= 0 || Frame::cosTheta(bRec.wi) <= 0)
		return 0.0f;

	float probSpecular = hasSpecular ? 1.0f : 0.0f;
	if (hasSpecular && hasDiffuse) {
		float Fi = MonteCarlo::fresnelDielectricExt(Frame::cosTheta(bRec.wi), m_eta);
		probSpecular = (Fi*m_specularSamplingWeight) /
			(Fi*m_specularSamplingWeight +
			(1-Fi) * (1-m_specularSamplingWeight));
	}

	if (hasSpecular && measure == EDiscrete) {
		/* Check if the provided direction pair matches an ideal
			specular reflection; tolerate some roundoff errors */
		if (abs(dot(reflect(bRec.wi), bRec.wo)-1) < DeltaEpsilon)
			return probSpecular;
	} else if (hasDiffuse && measure == ESolidAngle) {
		return Warp::squareToCosineHemispherePdf(bRec.wo) * (1-probSpecular);
	}

	return 0.0f;
}

Spectrum roughplastic::sample(BSDFSamplingRecord &bRec, float &_pdf, const float2 &_sample) const
{
	bool hasSpecular = (bRec.typeMask & EGlossyReflection);
	bool hasDiffuse = (bRec.typeMask & EDiffuseReflection);

	if (Frame::cosTheta(bRec.wi) <= 0 || (!hasSpecular && !hasDiffuse))
		return Spectrum(0.0f);
	
	bool choseSpecular = hasSpecular;
	float2 sample = _sample;

	/* Evaluate the roughness texture */
	float alpha = m_alpha.Evaluate(bRec.map).average();
	float alphaT = m_distribution.transformRoughness(alpha);

	float probSpecular;
	if (hasSpecular && hasDiffuse) {
		/* Find the probability of sampling the specular component */
		probSpecular = 1 - e_RoughTransmittanceManager::Evaluate(m_distribution.m_type, Frame::cosTheta(bRec.wi), alpha, m_eta);//m_externalRoughTransmittance->eval(Frame::cosTheta(bRec.wi), alpha)
		/* Reallocate samples */
		probSpecular = (probSpecular*m_specularSamplingWeight) /
			(probSpecular*m_specularSamplingWeight +
			(1-probSpecular) * (1-m_specularSamplingWeight));

		if (sample.y < probSpecular) {
			sample.y /= probSpecular;
		} else {
			sample.y = (sample.y - probSpecular) / (1 - probSpecular);
			choseSpecular = false;
		}
	}

	if (choseSpecular) {
		/* Perfect specular reflection based on the microsurface normal */
		float3 m = m_distribution.sample(sample, alphaT);
		bRec.wo = reflect(bRec.wi, m);
		bRec.sampledType = EGlossyReflection;

		/* Side check */
		if (Frame::cosTheta(bRec.wo) <= 0)
			return Spectrum(0.0f);
	} else {
		bRec.sampledType = EDiffuseReflection;
		bRec.wo = Warp::squareToCosineHemisphere(sample);
	}
	bRec.eta = 1.0f;

	/* Guard against numerical imprecisions */
	_pdf = pdf(bRec, ESolidAngle);

	if (_pdf == 0)
		return Spectrum(0.0f);
	else
		return f(bRec, ESolidAngle) / _pdf;
}

Spectrum roughplastic::f(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	bool hasSpecular = (bRec.typeMask & EGlossyReflection);
	bool hasDiffuse = (bRec.typeMask & EDiffuseReflection);

	if (measure != ESolidAngle ||
			Frame::cosTheta(bRec.wi) <= 0 ||
			Frame::cosTheta(bRec.wo) <= 0 ||
			(!hasSpecular && !hasDiffuse))
			return Spectrum(0.0f);

	/* Evaluate the roughness texture */
	float alpha = m_alpha.Evaluate(bRec.map).average();
	float alphaT = m_distribution.transformRoughness(alpha);

	Spectrum result(0.0f);
	if (hasSpecular)
	{
		/* Calculate the reflection half-vector */
		const float3 H = normalize(bRec.wo+bRec.wi);

		/* Evaluate the microsurface normal distribution */
		const float D = m_distribution.eval(H, alphaT);

		/* Fresnel term */
		const float F = MonteCarlo::fresnelDielectricExt(dot(bRec.wi, H), m_eta);

		/* Smith's shadow-masking function */
		const float G = m_distribution.G(bRec.wi, bRec.wo, H, alphaT);

		/* Calculate the specular reflection component */
		float value = F * D * G /
			(4.0f * Frame::cosTheta(bRec.wi));

		result += m_specularReflectance.Evaluate(bRec.map) * value;
	}

	if (hasDiffuse) {
		Spectrum diff = m_diffuseReflectance.Evaluate(bRec.map);
		//Float T12 = m_externalRoughTransmittance->eval(Frame::cosTheta(bRec.wi), alpha);
		//Float T21 = m_externalRoughTransmittance->eval(Frame::cosTheta(bRec.wo), alpha);
		//Float Fdr = 1-m_internalRoughTransmittance->evalDiffuse(alpha);
		float T12 = e_RoughTransmittanceManager::Evaluate(m_distribution.m_type, Frame::cosTheta(bRec.wi), alpha, m_eta);
		float T21 = e_RoughTransmittanceManager::Evaluate(m_distribution.m_type, Frame::cosTheta(bRec.wo), alpha, m_eta);
		float Fdr = e_RoughTransmittanceManager::EvaluateDiffuse(m_distribution.m_type, alpha, m_eta);

		if (m_nonlinear)
			diff /= Spectrum(1.0f) - diff * Fdr;
		else
			diff /= 1-Fdr;

		result += diff * (INV_PI * Frame::cosTheta(bRec.wo) * T12 * T21 * m_invEta2);
	}

	return result;
}

float roughplastic::pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	bool hasSpecular = (bRec.typeMask & EGlossyReflection);
	bool hasDiffuse = (bRec.typeMask & EDiffuseReflection);

	if (measure != ESolidAngle ||
		Frame::cosTheta(bRec.wi) <= 0 ||
		Frame::cosTheta(bRec.wo) <= 0 ||
		(!hasSpecular && !hasDiffuse))
		return 0.0f;

	/* Evaluate the roughness texture */
	float alpha = m_alpha.Evaluate(bRec.map).average();
	float alphaT = m_distribution.transformRoughness(alpha);

	/* Calculate the reflection half-vector */
	const float3 H = normalize(bRec.wo+bRec.wi);

	float probDiffuse, probSpecular;
	if (hasSpecular && hasDiffuse) {
		/* Find the probability of sampling the specular component */
		probSpecular = 1-e_RoughTransmittanceManager::Evaluate(m_distribution.m_type, Frame::cosTheta(bRec.wi), alpha, m_eta);

		/* Reallocate samples */
		probSpecular = (probSpecular*m_specularSamplingWeight) /
			(probSpecular*m_specularSamplingWeight +
			(1-probSpecular) * (1-m_specularSamplingWeight));

		probDiffuse = 1 - probSpecular;
	} else {
		probDiffuse = probSpecular = 1.0f;
	}

	float result = 0.0f;
	if (hasSpecular) {
		/* Jacobian of the half-direction mapping */
		const float dwh_dwo = 1.0f / (4.0f * dot(bRec.wo, H));

		/* Evaluate the microsurface normal distribution */
		const float prob = m_distribution.pdf(H, alphaT);

		result = prob * dwh_dwo * probSpecular;
	}

	if (hasDiffuse)
		result += probDiffuse * Warp::squareToCosineHemispherePdf(bRec.wo);

	return result;
}

Spectrum phong::sample(BSDFSamplingRecord &bRec, float &_pdf, const float2& _sample) const
{
	float2 sample = _sample;
	bool hasSpecular = (bRec.typeMask & EGlossyReflection);
	bool hasDiffuse  = (bRec.typeMask & EDiffuseReflection);

	if (!hasSpecular && !hasDiffuse)
		return Spectrum(0.0f);

	bool choseSpecular = hasSpecular;

	if (hasDiffuse && hasSpecular) {
		if (sample.x <= m_specularSamplingWeight) {
			sample.x /= m_specularSamplingWeight;
		} else {
			sample.x = (sample.x - m_specularSamplingWeight)
				/ (1-m_specularSamplingWeight);
			choseSpecular = false;
		}
	}

	if (choseSpecular) {
		float3 R = reflect(bRec.wi);
		float exponent = m_exponent.Evaluate(bRec.map).average();

		/* Sample from a Phong lobe centered around (0, 0, 1) */
		float sinAlpha = sqrtf(1-std::pow(sample.y, 2/(exponent + 1)));
		float cosAlpha = powf(sample.y, 1/(exponent + 1));
		float phi = (2.0f * PI) * sample.x;
		float3 localDir = make_float3(
			sinAlpha * std::cos(phi),
			sinAlpha * std::sin(phi),
			cosAlpha
		);

		/* Rotate into the correct coordinate system */
		bRec.wo = Frame(R).toWorld(localDir);
		bRec.sampledType = EGlossyReflection;

		if (Frame::cosTheta(bRec.wo) <= 0)
			return make_float3(0.0f);
	} else {
		bRec.wo = Warp::squareToCosineHemisphere(sample);
		bRec.sampledType = EDiffuseReflection;
	}
	bRec.eta = 1.0f;

	_pdf = pdf(bRec, ESolidAngle);

	if (_pdf == 0)
		return Spectrum(0.0f);
	else
		return f(bRec, ESolidAngle) / _pdf;
}

Spectrum phong::f(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	if (Frame::cosTheta(bRec.wi) <= 0 ||
		Frame::cosTheta(bRec.wo) <= 0 || measure != ESolidAngle)
		return Spectrum(0.0f);

	bool hasSpecular = (bRec.typeMask & EGlossyReflection);
	bool hasDiffuse  = (bRec.typeMask & EDiffuseReflection);

	Spectrum result = make_float3(0.0f);
	if (hasSpecular) {
		float alpha    = dot(bRec.wo, reflect(bRec.wi)),
			exponent = m_exponent.Evaluate(bRec.map).average();

		if (alpha > 0.0f) {
			result += m_specularReflectance.Evaluate(bRec.map) *
				((exponent + 2) * INV_TWOPI * std::pow(alpha, exponent));
		}
	}

	if (hasDiffuse)
		result += m_diffuseReflectance.Evaluate(bRec.map) * INV_PI;

	return result * Frame::cosTheta(bRec.wo);
}

float phong::pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	if (Frame::cosTheta(bRec.wi) <= 0 ||
		Frame::cosTheta(bRec.wo) <= 0 || measure != ESolidAngle)
		return 0.0f;

	bool hasSpecular = (bRec.typeMask & EGlossyReflection);
	bool hasDiffuse  = (bRec.typeMask & EDiffuseReflection);

	float diffuseProb = 0.0f, specProb = 0.0f;

	if (hasDiffuse)
		diffuseProb = Warp::squareToCosineHemispherePdf(bRec.wo);

	if (hasSpecular) {
		float alpha    = dot(bRec.wo, reflect(bRec.wi)),
			exponent = m_exponent.Evaluate(bRec.map).average();
		if (alpha > 0)
			specProb = std::pow(alpha, exponent) *
				(exponent + 1.0f) / (2.0f * PI);
	}

	if (hasDiffuse && hasSpecular)
		return m_specularSamplingWeight * specProb +
				(1-m_specularSamplingWeight) * diffuseProb;
	else if (hasDiffuse)
		return diffuseProb;
	else if (hasSpecular)
		return specProb;
	else
		return 0.0f;		
}

Spectrum ward::sample(BSDFSamplingRecord &bRec, float &_pdf, const float2 &_sample) const
{
	float2 sample = _sample;

	bool hasSpecular = (bRec.typeMask & EGlossyReflection);
	bool hasDiffuse  = (bRec.typeMask & EDiffuseReflection);

	if (!hasSpecular && !hasDiffuse)
		return Spectrum(0.0f);

	bool choseSpecular = hasSpecular;

	if (hasDiffuse && hasSpecular) {
		if (sample.x <= m_specularSamplingWeight) {
			sample.x /= m_specularSamplingWeight;
		} else {
			sample.x = (sample.x - m_specularSamplingWeight)
				/ (1-m_specularSamplingWeight);
			choseSpecular = false;
		}
	}

	if (choseSpecular) {
		float alphaU = m_alphaU.Evaluate(bRec.map).average();
		float alphaV = m_alphaV.Evaluate(bRec.map).average();
			
		float phiH = std::atan(alphaV/alphaU
			* std::tan(2.0f * PI * sample.y));
		if (sample.y > 0.5f)
			phiH += PI;
		float cosPhiH = cosf(phiH);
		float sinPhiH = math::safe_sqrt(1.0f-cosPhiH*cosPhiH);
			
		float thetaH = atan(math::safe_sqrt(
			-logf(sample.x) / (
				(cosPhiH*cosPhiH) / (alphaU*alphaU) +
				(sinPhiH*sinPhiH) / (alphaV*alphaV)
		)));
		float3 H = MonteCarlo::SphericalDirection(thetaH, phiH);
		bRec.wo = H * (2.0f * dot(bRec.wi, H)) - bRec.wi;

		bRec.sampledType = EGlossyReflection;

		if (Frame::cosTheta(bRec.wo) <= 0.0f)
			return Spectrum(0.0f);
	} else {
		bRec.wo = Warp::squareToCosineHemisphere(sample);
		bRec.sampledType = EDiffuseReflection;
	}
	bRec.eta = 1.0f;

	_pdf = pdf(bRec, ESolidAngle);

	if (_pdf == 0)
		return Spectrum(0.0f);
	else
		return f(bRec, ESolidAngle) / _pdf;
}

Spectrum ward::f(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	if (Frame::cosTheta(bRec.wi) <= 0 ||
		Frame::cosTheta(bRec.wo) <= 0 || measure != ESolidAngle)
		return Spectrum(0.0f);

	bool hasSpecular = (bRec.typeMask & EGlossyReflection);
	bool hasDiffuse  = (bRec.typeMask & EDiffuseReflection);

	Spectrum result = Spectrum(0.0f);
	if (hasSpecular) {
		float3 H = bRec.wi+bRec.wo;
		float alphaU = m_alphaU.Evaluate(bRec.map).average();
		float alphaV = m_alphaV.Evaluate(bRec.map).average();
		float factor1 = 0.0f;
		switch (m_modelVariant) {
			case EWard:
				factor1 = 1.0f / (4.0f * PI * alphaU * alphaV *
					sqrtf(Frame::cosTheta(bRec.wi)*Frame::cosTheta(bRec.wo)));
				break;
			case EWardDuer:
				factor1 = 1.0f / (4.0f * PI * alphaU * alphaV *
					Frame::cosTheta(bRec.wi)*Frame::cosTheta(bRec.wo));
				break;
			case EBalanced:
				factor1 = dot(H,H) / (PI * alphaU * alphaV
					* std::pow(Frame::cosTheta(H),4));
				break;
		}

		float factor2 = H.x / alphaU, factor3 = H.y / alphaV;
		float exponent = -(factor2*factor2+factor3*factor3)/(H.z*H.z);
		float specRef = factor1 * exp(exponent);
		/* Important to prevent numeric issues when evaluating the
			sampling density of the Ward model in places where it takes
			on miniscule values (Veach-MLT does this for instance) */
		if (specRef > 1e-10f)
			result += m_specularReflectance.Evaluate(bRec.map) * specRef;
	}

	if (hasDiffuse)
		result += m_diffuseReflectance.Evaluate(bRec.map) * INV_PI;

	return result * Frame::cosTheta(bRec.wo);
}

float ward::pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	if (Frame::cosTheta(bRec.wi) <= 0 ||
		Frame::cosTheta(bRec.wo) <= 0 || measure != ESolidAngle)
		return 0.0f;

	bool hasSpecular = (bRec.typeMask & EGlossyReflection);
	bool hasDiffuse  = (bRec.typeMask & EDiffuseReflection);

	float diffuseProb = 0.0f, specProb = 0.0f;

	if (hasSpecular) {
		float alphaU = m_alphaU.Evaluate(bRec.map).average();
		float alphaV = m_alphaV.Evaluate(bRec.map).average();
		float3 H = normalize(bRec.wi+bRec.wo);
		float factor1 = 1.0f / (4.0f * PI * alphaU * alphaV *
			dot(H, bRec.wi) * powf(Frame::cosTheta(H), 3));
		float factor2 = H.x / alphaU, factor3 = H.y / alphaV;

		float exponent = -(factor2*factor2+factor3*factor3)/(H.z*H.z);
		specProb = factor1 * exp(exponent);
	}

	if (hasDiffuse)
		diffuseProb = Warp::squareToCosineHemispherePdf(bRec.wo);

	if (hasDiffuse && hasSpecular)
		return m_specularSamplingWeight * specProb +
				(1-m_specularSamplingWeight) * diffuseProb;
	else if (hasDiffuse)
		return diffuseProb;
	else if (hasSpecular)
		return specProb;
	else
		return 0.0f;
}

Spectrum hk::sample(BSDFSamplingRecord &bRec, float &_pdf, const float2 &_sample) const
{
	bool hasSpecularTransmission = (bRec.typeMask & EDeltaTransmission);
	bool hasSingleScattering = (bRec.typeMask & EGlossy);

	const Spectrum sigmaA = m_sigmaA.Evaluate(bRec.map),
				sigmaS = m_sigmaS.Evaluate(bRec.map),
				sigmaT = sigmaA + sigmaS,
				tauD = sigmaT * m_thickness;

	/* Probability for a specular transmission is approximated by the average (per wavelength)
		* probability of a photon exiting without a scattering event or an absorption event */
	float probSpecularTransmission = ((-1.0f * tauD/abs(Frame::cosTheta(bRec.wi))).exp()).average();

	bool choseSpecularTransmission = hasSpecularTransmission;

	float2 sample = (_sample);
	if (hasSpecularTransmission && hasSingleScattering) {
		if (sample.x > probSpecularTransmission) {
			sample.x = (sample.x - probSpecularTransmission) / (1 - probSpecularTransmission);
			choseSpecularTransmission = false;
		}
	}

	bRec.eta = 1.0f;
	if (choseSpecularTransmission) {
		/* The specular transmission component was sampled */
		bRec.sampledType = EDeltaTransmission;

		bRec.wo = -bRec.wi;

		_pdf = hasSingleScattering ? probSpecularTransmission : 1.0f;
		return f(bRec, EDiscrete) / _pdf;
	} else {
		/* The glossy transmission/scattering component should be sampled */
		bool hasGlossyReflection = (bRec.typeMask & EGlossyReflection);
		bool hasGlossyTransmission = (bRec.typeMask & EGlossyTransmission);

		/* Sample According to the phase function lobes */
		PhaseFunctionSamplingRecord pRec(bRec.wi, bRec.wo);
		m_phase.Sample(pRec, _pdf, *bRec.rng);

		/* Store the sampled direction */
		bRec.wo = pRec.wo;

		bool reflection = Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) >= 0;
		if ((!hasGlossyReflection && reflection) ||
			(!hasGlossyTransmission && !reflection))
			return Spectrum(0.0f);

		/* Notify that the scattering component was sampled */
		bRec.sampledType = EGlossy;

		_pdf *= (hasSpecularTransmission ? (1 - probSpecularTransmission) : 1.0f);

		/* Guard against numerical imprecisions */
		if (_pdf == 0)
			return Spectrum(0.0f);
		else
			return f(bRec, ESolidAngle) / _pdf;

	}
}

Spectrum hk::f(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	Spectrum sigmaA = m_sigmaA.Evaluate(bRec.map),
				sigmaS = m_sigmaS.Evaluate(bRec.map),
				sigmaT = sigmaA + sigmaS,
				tauD = sigmaT * m_thickness,
				result = Spectrum(0.0f);

	if (measure == EDiscrete) {
		/* Figure out if the specular transmission is specifically requested */
		bool hasSpecularTransmission = (bRec.typeMask & EDeltaTransmission);

		/* Return the attenuated light if requested */
		if (hasSpecularTransmission &&
			abs(1+dot(bRec.wi, bRec.wo)) < DeltaEpsilon)
			result = (-tauD/abs(Frame::cosTheta(bRec.wi))).exp();
	} else if (measure == ESolidAngle) {
		/* Sample single scattering events */
		bool hasGlossyReflection = (bRec.typeMask & EGlossyReflection);
		bool hasGlossyTransmission = (bRec.typeMask & EGlossyTransmission);

		Spectrum albedo;
			for (int i = 0; i < SPECTRUM_SAMPLES; i++)
				albedo[i] = sigmaT[i] > 0 ? (sigmaS[i]/sigmaT[i]) : (float) 0;

		const float cosThetaI = Frame::cosTheta(bRec.wi),
				    cosThetaO = Frame::cosTheta(bRec.wo),
				    dp = cosThetaI*cosThetaO;

		bool reflection = dp > 0, transmission = dp < 0;

		/* ==================================================================== */
		/*                        Reflection component                          */
		/* ==================================================================== */

		if (hasGlossyReflection && reflection) {
			PhaseFunctionSamplingRecord pRec(bRec.wi,bRec.wo);
			const float phaseVal = m_phase.Evaluate(pRec);

			result = albedo * (phaseVal*cosThetaI/(cosThetaI+cosThetaO)) *
				(Spectrum(1.0f)-((-1.0f/abs(cosThetaI)-1.0f/abs(cosThetaO)) * tauD).exp());
		}

		/* ==================================================================== */
		/*                       Transmission component                         */
		/* ==================================================================== */

		if (hasGlossyTransmission && transmission
			&& m_thickness < FLT_MAX) {
			PhaseFunctionSamplingRecord pRec(bRec.wi,bRec.wo);
			const float phaseVal = m_phase.Evaluate(pRec);

			/* Hanrahan etal 93 Single Scattering transmission term */
			if (abs(cosThetaI + cosThetaO) < EPSILON) {
				/* avoid division by zero */
				result += albedo * phaseVal*tauD/abs(cosThetaO) *
					((-tauD/abs(cosThetaO)).exp());
			} else {
				/* Guaranteed to be positive even if |cosThetaO| > |cosThetaI| */
				result += albedo * phaseVal*abs(cosThetaI)/(abs(cosThetaI)-abs(cosThetaO)) *
					((-tauD/abs(cosThetaI)).exp() - (-tauD/abs(cosThetaO)).exp());
			}
		}
		return result * abs(cosThetaO);
	}
	return result;
}

float hk::pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	bool hasSingleScattering = (bRec.typeMask & EGlossy);
	bool hasSpecularTransmission = (bRec.typeMask & EDeltaTransmission);

	const Spectrum sigmaA = m_sigmaA.Evaluate(bRec.map),
				sigmaS = m_sigmaS.Evaluate(bRec.map),
				sigmaT = sigmaA + sigmaS,
				tauD = sigmaT * m_thickness;

	float probSpecularTransmission = ((-1.0f * tauD/abs(Frame::cosTheta(bRec.wi))).exp()).average();

	if (measure == EDiscrete) {
		bool hasSpecularTransmission = (bRec.typeMask & EDeltaTransmission);
		/* Return the attenuated light if requested */
		if (hasSpecularTransmission &&
			abs(1+dot(bRec.wi, bRec.wo)) < DeltaEpsilon)
			return hasSingleScattering ? probSpecularTransmission : 1.0f;
	} else if (hasSingleScattering && measure == ESolidAngle) {
		bool hasGlossyReflection = (bRec.typeMask & EGlossyReflection);
		bool hasGlossyTransmission = (bRec.typeMask & EGlossyTransmission);
		bool reflection = Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) >= 0;

		if ((!hasGlossyReflection && reflection) ||
			(!hasGlossyTransmission && !reflection))
			return 0.0f;

		/* Sampled according to the phase function lobe(s) */
		PhaseFunctionSamplingRecord pRec(bRec.wi, bRec.wo);
		float pdf = m_phase.pdf(pRec);
		if (hasSpecularTransmission)
			pdf *= 1-probSpecularTransmission;
		return pdf;
	}
	return 0.0f;
}