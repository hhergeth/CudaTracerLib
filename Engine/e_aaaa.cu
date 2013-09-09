#include "e_BSDF.h"

Spectrum roughcoating::sample(BSDFSamplingRecord &bRec, float &_pdf, const float2 &_sample) const
{
	bRec.sampledType = EDeltaReflection;
	float F = MonteCarlo::fresnelDielectricExt(Frame::cosTheta(bRec.wi), m_eta);
	bRec.wo = make_float3(-bRec.wi.x, -bRec.wi.y, bRec.wi.z);
	return F;

	bool hasNested = (bRec.typeMask & m_nested.getType() & EAll)
		&& (bRec.component == -1);
	bool hasSpecular = (bRec.typeMask & EGlossyReflection)
		&& (bRec.component == -1);

	bool choseSpecular = hasSpecular;
	float2 sample = _sample;

	/* Evaluate the roughness texture */
	float alpha = m_alpha.Evaluate(bRec.map);
	float alphaT = m_distribution.transformRoughness(alpha);

	float probSpecular;
	if (hasSpecular && hasNested) {
		/* Find the probability of sampling the diffuse component */
		probSpecular = 1 - e_RoughTransmittanceManager::Evaluate(m_distribution.m_type, Frame::cosTheta(bRec.wi), alpha, m_eta);

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
		bRec.sampledComponent = 2;//BUG
		bRec.sampledType = EGlossyReflection;
		bRec.eta = 1.0f;
		
		/* Side check */
		if (Frame::cosTheta(bRec.wo) * Frame::cosTheta(bRec.wi) <= 0)
			return Spectrum(0.0f);
	} else {
		float3 wiBackup = bRec.wi;
		bRec.wi = refractTo(EInterior, bRec.wi);
		Spectrum result = m_nested.sample(bRec, _pdf, sample);
		bRec.wi = wiBackup;
		if (result.isZero())
			return Spectrum(0.0f);
		bRec.wo = refractTo(EExterior, bRec.wo);
		if (dot(bRec.wo, bRec.wo) == 0.0f)
			return Spectrum(0.0f);
	}

	/* Guard against numerical imprecisions */
	EMeasure measure = getMeasure(bRec.sampledType);
	_pdf = pdf(bRec, measure);

	if (_pdf == 0)
		return Spectrum(0.0f);
	else
		return f(bRec, measure) / _pdf;
}

Spectrum roughcoating::f(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	bool hasNested = (bRec.typeMask & m_nested.getType() & EAll)
		&& (bRec.component == -1);
	bool hasSpecular = (bRec.typeMask & EGlossyReflection)
		&& (bRec.component == -1)
		&& measure == ESolidAngle;

	/* Evaluate the roughness texture */
	float alpha = m_alpha.Evaluate(bRec.map);
	float alphaT = m_distribution.transformRoughness(alpha);

	Spectrum result(0.0f);
	if (hasSpecular && Frame::cosTheta(bRec.wo) * Frame::cosTheta(bRec.wi) > 0) {
		/* Calculate the reflection half-vector */
		const float3 H = normalize(bRec.wo+bRec.wi)
			* math::signum(Frame::cosTheta(bRec.wo));

		/* Evaluate the microsurface normal distribution */
		const float D = m_distribution.eval(H, alphaT);

		/* Fresnel term */
		const float F = MonteCarlo::fresnelDielectricExt(AbsDot(bRec.wi, H), m_eta);

		/* Smith's shadow-masking function */
		const float G = m_distribution.G(bRec.wi, bRec.wo, H, alphaT);

		/* Calculate the specular reflection component */
		float value = F * D * G /
			(4.0f * abs(Frame::cosTheta(bRec.wi)));

		result += m_specularReflectance.Evaluate(bRec.map) * value;
	}

	if (hasNested) {
		BSDFSamplingRecord bRecInt(bRec);
		bRecInt.wi = refractTo(EInterior, bRec.wi);
		bRecInt.wo = refractTo(EInterior, bRec.wo);

		Spectrum nestedResult = m_nested.f(bRecInt, measure) *
			e_RoughTransmittanceManager::Evaluate(m_distribution.m_type, Frame::cosTheta(bRec.wi), alpha, m_eta) *
			e_RoughTransmittanceManager::Evaluate(m_distribution.m_type, Frame::cosTheta(bRec.wo), alpha, m_eta);

		Spectrum sigmaA = m_sigmaA.Evaluate(bRec.map) * m_thickness;
		if (!sigmaA.isZero())
			nestedResult *= (-sigmaA *
				(1/abs(Frame::cosTheta(bRecInt.wi)) +
					1/abs(Frame::cosTheta(bRecInt.wo)))).exp();

		if (measure == ESolidAngle) {
			/* Solid angle compression & irradiance conversion factors */
			nestedResult *= m_invEta * m_invEta *
					Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo)
				/ (Frame::cosTheta(bRecInt.wi) * Frame::cosTheta(bRecInt.wo));
		}

		result += nestedResult;
	}

	return result;
}

float roughcoating::pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	bool hasNested = (bRec.typeMask & m_nested.getType() & EAll)
		&& (bRec.component == -1);
	bool hasSpecular = (bRec.typeMask & EGlossyReflection)
		&& (bRec.component == -1)
		&& measure == ESolidAngle;

	/* Calculate the reflection half-vector */
	const float3 H = normalize(bRec.wo+bRec.wi)
			* math::signum(Frame::cosTheta(bRec.wo));

	/* Evaluate the roughness texture */
	float alpha = m_alpha.Evaluate(bRec.map);
	float alphaT = m_distribution.transformRoughness(alpha);

	float probNested, probSpecular;
	if (hasSpecular && hasNested) {
		/* Find the probability of sampling the specular component */
		probSpecular = 1-e_RoughTransmittanceManager::Evaluate(m_distribution.m_type, Frame::cosTheta(bRec.wi), alpha, m_eta);

		/* Reallocate samples */
		probSpecular = (probSpecular*m_specularSamplingWeight) /
			(probSpecular*m_specularSamplingWeight +
			(1-probSpecular) * (1-m_specularSamplingWeight));

		probNested = 1 - probSpecular;
	} else {
		probNested = probSpecular = 1.0f;
	}

	float result = 0.0f;
	if (hasSpecular && Frame::cosTheta(bRec.wo) * Frame::cosTheta(bRec.wi) > 0) {
		/* Jacobian of the half-direction mapping */
		const float dwh_dwo = 1.0f / (4.0f * AbsDot(bRec.wo, H));

		/* Evaluate the microsurface normal distribution */
		const float prob = m_distribution.pdf(H, alphaT);

		result = prob * dwh_dwo * probSpecular;
	}

	if (hasNested) {
		BSDFSamplingRecord bRecInt(bRec);
		bRecInt.wi = refractTo(EInterior, bRec.wi);
		bRecInt.wo = refractTo(EInterior, bRec.wo);

		float prob = m_nested.pdf(bRecInt, measure);

		if (measure == ESolidAngle) {
			prob *= m_invEta * m_invEta * Frame::cosTheta(bRec.wo)
			        / Frame::cosTheta(bRecInt.wo);
		}

		result += prob * probNested;
	}

	return result;
}