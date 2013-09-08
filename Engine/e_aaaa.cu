#include "e_BSDF.h"

Spectrum roughplastic::sample(BSDFSamplingRecord &bRec, float &_pdf, const float2 &_sample) const
{
	bool hasSpecular = (bRec.typeMask & EGlossyReflection) &&
		(bRec.component == -1 || bRec.component == 0);
	bool hasDiffuse = (bRec.typeMask & EDiffuseReflection) &&
		(bRec.component == -1 || bRec.component == 1);

	if (Frame::cosTheta(bRec.wi) <= 0 || (!hasSpecular && !hasDiffuse))
		return Spectrum(0.0f);
	
	bool choseSpecular = hasSpecular;
	float2 sample = _sample;

	/* Evaluate the roughness texture */
	float alpha = m_alpha.Evaluate(bRec.map);
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
		bRec.sampledComponent = 0;
		bRec.sampledType = EGlossyReflection;

		/* Side check */
		if (Frame::cosTheta(bRec.wo) <= 0)
			return Spectrum(0.0f);
	} else {
		bRec.sampledComponent = 1;
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
	bool hasSpecular = (bRec.typeMask & EGlossyReflection) &&
			(bRec.component == -1 || bRec.component == 0);
	bool hasDiffuse = (bRec.typeMask & EDiffuseReflection) &&
			(bRec.component == -1 || bRec.component == 1);

	if (measure != ESolidAngle ||
			Frame::cosTheta(bRec.wi) <= 0 ||
			Frame::cosTheta(bRec.wo) <= 0 ||
			(!hasSpecular && !hasDiffuse))
			return Spectrum(0.0f);

	/* Evaluate the roughness texture */
	float alpha = m_alpha.Evaluate(bRec.map);
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
	bool hasSpecular = (bRec.typeMask & EGlossyReflection) &&
		(bRec.component == -1 || bRec.component == 0);
	bool hasDiffuse = (bRec.typeMask & EDiffuseReflection) &&
		(bRec.component == -1 || bRec.component == 1);

	if (measure != ESolidAngle ||
		Frame::cosTheta(bRec.wi) <= 0 ||
		Frame::cosTheta(bRec.wo) <= 0 ||
		(!hasSpecular && !hasDiffuse))
		return 0.0f;

	/* Evaluate the roughness texture */
	float alpha = m_alpha.Evaluate(bRec.map);
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