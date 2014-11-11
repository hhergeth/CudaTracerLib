#include "e_BSDF.h"

Spectrum dielectric::sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const
{
	bool sampleReflection = (bRec.typeMask & EDeltaReflection);
	bool sampleTransmission = (bRec.typeMask & EDeltaTransmission);

	float cosThetaT;
	float F = MonteCarlo::fresnelDielectricExt(Frame::cosTheta(bRec.wi), cosThetaT, m_eta);

	if (sampleTransmission && sampleReflection) {
		//float f0 = m_specularReflectance.Evaluate(bRec.map).average(), f1 = m_specularTransmittance.Evaluate(bRec.map).average(), f = F*f0/(f0+f1);
		if (sample.x <= F) {
			bRec.sampledType = EDeltaReflection;
			bRec.wo = reflect(bRec.wi);
			bRec.eta = 1.0f;
			pdf = F;

			return m_specularReflectance.Evaluate(bRec.dg);// / f * F;
		}
		else {
			bRec.sampledType = EDeltaTransmission;
			bRec.wo = refract(bRec.wi, cosThetaT);
			bRec.eta = cosThetaT < 0 ? m_eta : m_invEta;
			pdf = 1 - F;

			float factor = (bRec.mode == ERadiance)
				? (cosThetaT < 0 ? m_invEta : m_eta) : 1.0f;

			return m_specularTransmittance.Evaluate(bRec.dg) * (factor * factor);// / (1 - f) * (1 - F);
		}
	}
	else if (sampleReflection) {
		bRec.sampledType = EDeltaReflection;
		bRec.wo = reflect(bRec.wi);
		bRec.eta = 1.0f;
		pdf = 1.0f;

		return m_specularReflectance.Evaluate(bRec.dg);
	}
	else if (sampleTransmission) {
		bRec.sampledType = EDeltaTransmission;
		bRec.wo = refract(bRec.wi, cosThetaT);
		bRec.eta = cosThetaT < 0 ? m_eta : m_invEta;
		pdf = 1.0f;

		float factor = (bRec.mode == ERadiance)
			? (cosThetaT < 0 ? m_invEta : m_eta) : 1.0f;

		return m_specularTransmittance.Evaluate(bRec.dg) * (factor * factor * (1 - F));
	}

	return Spectrum(0.0f);
}
