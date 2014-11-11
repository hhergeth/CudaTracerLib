#include "e_BSDF.h"

Spectrum coating::sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &_sample) const
{
	bool sampleSpecular = (bRec.typeMask & EDeltaReflection);
	bool sampleNested = (bRec.typeMask & m_nested.getType() & EAll);

	if ((!sampleSpecular && !sampleNested))
		return Spectrum(0.0f);

	float R12;
	float3 wiPrime = refractIn(bRec.wi, R12);

	float probSpecular = (R12*m_specularSamplingWeight) /
		(R12*m_specularSamplingWeight +
		(1-R12) * (1-m_specularSamplingWeight));

	bool choseSpecular = sampleSpecular;

	float2 sample = _sample;
	if (sampleSpecular && sampleNested) {
		if (sample.x < probSpecular) {
			sample.x /= probSpecular;
		} else {
			sample.x = (sample.x - probSpecular) / (1 - probSpecular);
			choseSpecular = false;
		}
	}

	if (choseSpecular) {
		bRec.sampledType = EDeltaReflection;
		bRec.wo = reflect(bRec.wi);
		bRec.eta = 1.0f;
		pdf = sampleNested ? probSpecular : 1.0f;
		return m_specularReflectance.Evaluate(bRec.map) * (R12/pdf);
	} else {
		if (R12 == 1.0f) 
			return Spectrum(0.0f);

		float3 wiBackup = bRec.wi;
		bRec.wi = wiPrime;
		Spectrum result = m_nested.sample(bRec, pdf, sample);
		bRec.wi = wiBackup;
		if (result.isZero())
			return Spectrum(0.0f);

		float3 woPrime = bRec.wo;

		Spectrum sigmaA = m_sigmaA.Evaluate(bRec.map) * m_thickness;
		if (!sigmaA.isZero())
			result *= (-sigmaA *
				(1/std::abs(Frame::cosTheta(wiPrime)) +
					1/std::abs(Frame::cosTheta(woPrime)))).exp();

		float R21;
		bRec.wo = refractOut(woPrime, R21);
		if (R21 == 1.0f) 
			return Spectrum(0.0f);

		if (sampleSpecular) {
			pdf *= 1.0f - probSpecular;
			result /= 1.0f - probSpecular;
		}

		result *= (1 - R12) * (1 - R21);

		if (BSDF::getMeasure(bRec.sampledType) == ESolidAngle) {
			result *= Frame::cosTheta(bRec.wi) / Frame::cosTheta(wiPrime);
			pdf *= m_invEta * m_invEta * Frame::cosTheta(bRec.wo) / Frame::cosTheta(woPrime);
		}

		return result;
	}
}

Spectrum coating::f(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	bool sampleSpecular = (bRec.typeMask & EDeltaReflection);
	bool sampleNested = (bRec.typeMask & m_nested.getType() & EAll);

	if (measure == EDiscrete && sampleSpecular &&
			std::abs(dot(reflect(bRec.wi), bRec.wo)-1) < DeltaEpsilon) {
		return m_specularReflectance.Evaluate(bRec.map) *
			MonteCarlo::fresnelDielectricExt(abs(Frame::cosTheta(bRec.wi)), m_eta);
	} else if (sampleNested) {
		float R12, R21;
		BSDFSamplingRecord bRecInt(bRec);
		bRecInt.wi = refractIn(bRec.wi, R12);
		bRecInt.wo = refractIn(bRec.wo, R21);

		if (R12 == 1 || R21 == 1)
			return Spectrum(0.0f);

		Spectrum result = m_nested.f(bRecInt, measure)
			* (1-R12) * (1-R21);

		Spectrum sigmaA = m_sigmaA.Evaluate(bRec.map) * m_thickness;
		if (!sigmaA.isZero())
			result *= (-sigmaA *
				(1/std::abs(Frame::cosTheta(bRecInt.wi)) +
					1/abs(Frame::cosTheta(bRecInt.wo)))).exp();

		if (measure == ESolidAngle) {
			result *= m_invEta * m_invEta *
					Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo)
				/ (Frame::cosTheta(bRecInt.wi) * Frame::cosTheta(bRecInt.wo));
		}

		return result;
	}

	return Spectrum(0.0f);
}

float coating::pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	bool sampleSpecular = (bRec.typeMask & EDeltaReflection);
	bool sampleNested = (bRec.typeMask & m_nested.getType() & EAll);

	float R12;
	float3 wiPrime = refractIn(bRec.wi, R12);

	float probSpecular = (R12*m_specularSamplingWeight) /
		(R12*m_specularSamplingWeight +
		(1-R12) * (1-m_specularSamplingWeight));

	if (measure == EDiscrete && sampleSpecular &&
			abs(dot(reflect(bRec.wi), bRec.wo)-1) < DeltaEpsilon) {
		return sampleNested ? probSpecular : 1.0f;
	} else if (sampleNested) {
		float R21;
		BSDFSamplingRecord bRecInt(bRec);
		bRecInt.wi = wiPrime;
		bRecInt.wo = refractIn(bRec.wo, R21);

		if (R12 == 1 || R21 == 1)
			return 0.0f;

		float pdf = m_nested.pdf(bRecInt, measure);

		if (measure == ESolidAngle)
			pdf *= m_invEta * m_invEta * Frame::cosTheta(bRec.wo)
			        / Frame::cosTheta(bRecInt.wo);

		return sampleSpecular ? (pdf * (1 - probSpecular)) : pdf;
	} else {
		return 0.0f;
	}
}

Spectrum roughcoating::sample(BSDFSamplingRecord &bRec, float &_pdf, const float2 &_sample) const
{
	bool hasNested = (bRec.typeMask & m_nested.getType() & EAll);
	bool hasSpecular = (bRec.typeMask & EGlossyReflection);

	bool choseSpecular = hasSpecular;
	float2 sample = _sample;

	/* Evaluate the roughness texture */
	float alpha = m_alpha.Evaluate(bRec.map).average();
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
	bool hasNested = (bRec.typeMask & m_nested.getType() & EAll);
	bool hasSpecular = (bRec.typeMask & EGlossyReflection) && measure == ESolidAngle;

	/* Evaluate the roughness texture */
	float alpha = m_alpha.Evaluate(bRec.map).average();
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
	bool hasNested = (bRec.typeMask & m_nested.getType() & EAll);
	bool hasSpecular = (bRec.typeMask & EGlossyReflection) && measure == ESolidAngle;

	/* Calculate the reflection half-vector */
	const float3 H = normalize(bRec.wo+bRec.wi)
			* math::signum(Frame::cosTheta(bRec.wo));

	/* Evaluate the roughness texture */
	float alpha = m_alpha.Evaluate(bRec.map).average();
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

Spectrum blend::sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &_sample) const
{
	float weights[2];
	weights[1] = clamp(this->weight.Evaluate(bRec.map).average(), 0.0f, 1.0f);
	weights[0] = 1.0f - weights[1];

	float2 sample = _sample;
	unsigned int entry;
	if (sample.x < weights[0])
	{
		entry = 0; sample.x /= weights[0];
	}
	else { entry = 1; sample.x = (sample.x - weights[0]) / weights[1]; }
	Spectrum result = bsdfs[entry].sample(bRec, pdf, sample);
	if (result.isZero()) // sampling failed
		return result;

	result *= weights[entry] * pdf;
	pdf *= weights[entry];

	EMeasure measure = BSDF::getMeasure(bRec.sampledType);
	for (size_t i = 0; i<2; ++i) {
		if (entry == i)
			continue;
		pdf += bsdfs[i].pdf(bRec, measure) * weights[i];
		result += bsdfs[i].f(bRec, measure) * weights[i];
	}
	return result / pdf;
}

Spectrum blend::f(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	float weight = clamp(this->weight.Evaluate(bRec.map).average(), 0.0f, 1.0f);
	return bsdfs[0].f(bRec, measure) * (1-weight) + bsdfs[1].f(bRec, measure) * weight;
}

float blend::pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	float weight = clamp(this->weight.Evaluate(bRec.map).average(), 0.0f, 1.0f);
	return bsdfs[0].pdf(bRec, measure) * (1-weight) + bsdfs[1].pdf(bRec, measure) * weight;
}