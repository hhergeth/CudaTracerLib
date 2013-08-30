#include "e_BSDF.h"
/*
CUDA_FUNC_IN float3 coating::sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &sample) const
{
		bool sampleSpecular = (bRec.typeMask & EDeltaReflection)
			&& (bRec.component == -1 || bRec.component == (int) m_components.size()-1);
		bool sampleNested = (bRec.typeMask & m_nested->getType() & BSDF::EAll)
			&& (bRec.component == -1 || bRec.component < (int) m_components.size()-1);

		if ((!sampleSpecular && !sampleNested))
			return Spectrum(0.0f);

		Float R12;
		Vector wiPrime = refractIn(bRec.wi, R12);

		Float probSpecular = (R12*m_specularSamplingWeight) /
			(R12*m_specularSamplingWeight +
			(1-R12) * (1-m_specularSamplingWeight));

		bool choseSpecular = sampleSpecular;

		Point2 sample(_sample);
		if (sampleSpecular && sampleNested) {
			if (sample.x < probSpecular) {
				sample.x /= probSpecular;
			} else {
				sample.x = (sample.x - probSpecular) / (1 - probSpecular);
				choseSpecular = false;
			}
		}

		if (choseSpecular) {
			bRec.sampledComponent = (int) m_components.size() - 1;
			bRec.sampledType = EDeltaReflection;
			bRec.wo = reflect(bRec.wi);
			bRec.eta = 1.0f;
			pdf = sampleNested ? probSpecular : 1.0f;
			return m_specularReflectance->eval(bRec.its) * (R12/pdf);
		} else {
			if (R12 == 1.0f) 
				return Spectrum(0.0f);

			Vector wiBackup = bRec.wi;
			bRec.wi = wiPrime;
			Spectrum result = m_nested->sample(bRec, pdf, sample);
			bRec.wi = wiBackup;
			if (result.isZero())
				return Spectrum(0.0f);

			Vector woPrime = bRec.wo;

			Spectrum sigmaA = m_sigmaA->eval(bRec.its) * m_thickness;
			if (!sigmaA.isZero())
				result *= (-sigmaA *
					(1/std::abs(Frame::cosTheta(wiPrime)) +
					 1/std::abs(Frame::cosTheta(woPrime)))).exp();

			Float R21;
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

CUDA_FUNC_IN float3 coating::f(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
			bool sampleSpecular = (bRec.typeMask & EDeltaReflection)
			&& (bRec.component == -1 || bRec.component == (int) m_components.size()-1);
		bool sampleNested = (bRec.typeMask & m_nested->getType() & BSDF::EAll)
			&& (bRec.component == -1 || bRec.component < (int) m_components.size()-1);

		if (measure == EDiscrete && sampleSpecular &&
			    std::abs(dot(reflect(bRec.wi), bRec.wo)-1) < DeltaEpsilon) {
			return m_specularReflectance->eval(bRec.its) *
				fresnelDielectricExt(std::abs(Frame::cosTheta(bRec.wi)), m_eta);
		} else if (sampleNested) {
			Float R12, R21;
			BSDFSamplingRecord bRecInt(bRec);
			bRecInt.wi = refractIn(bRec.wi, R12);
			bRecInt.wo = refractIn(bRec.wo, R21);

			if (R12 == 1 || R21 == 1)
				return Spectrum(0.0f);

			Spectrum result = m_nested->eval(bRecInt, measure)
				* (1-R12) * (1-R21);

			Spectrum sigmaA = m_sigmaA->eval(bRec.its) * m_thickness;
			if (!sigmaA.isZero())
				result *= (-sigmaA *
					(1/std::abs(Frame::cosTheta(bRecInt.wi)) +
					 1/std::abs(Frame::cosTheta(bRecInt.wo)))).exp();

			if (measure == ESolidAngle) {
				result *= m_invEta * m_invEta *
					  Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo)
				   / (Frame::cosTheta(bRecInt.wi) * Frame::cosTheta(bRecInt.wo));
			}

			return result;
		}

		return Spectrum(0.0f);
}

CUDA_FUNC_IN float coating::pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const
{
	bool sampleSpecular = (bRec.typeMask & EDeltaReflection)
		&& (bRec.component == -1 || bRec.component == (int) m_components.size()-1);
	bool sampleNested = (bRec.typeMask & m_nested->getType() & BSDF::EAll)
		&& (bRec.component == -1 || bRec.component < (int) m_components.size()-1);

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

		float pdf = m_nested->pdf(bRecInt, measure);

		if (measure == ESolidAngle)
			pdf *= m_invEta * m_invEta * Frame::cosTheta(bRec.wo)
			        / Frame::cosTheta(bRecInt.wo);

		return sampleSpecular ? (pdf * (1 - probSpecular)) : pdf;
	} else {
		return 0.0f;
	}
}*/