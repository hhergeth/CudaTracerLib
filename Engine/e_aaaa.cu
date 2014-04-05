#include "e_BSDF.h"

Spectrum blend::sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &_sample) const
{
	float weights[2];
	weights[1] = clamp(this->weight.Evaluate(bRec.map).average(), 0.0f, 1.0f);
	weights[0] = 1.0f - weights[1];

	float2 sample = _sample;
	unsigned int entry;
	if (sample.x < weights[0])
		{entry = 0; sample.x /= weights[0]; }
	else { entry = 1; sample.x = (sample.x - weights[0]) / weights[1]; }
	Spectrum result = bsdfs[entry].sample(bRec, pdf, sample);
	if (result.isZero()) // sampling failed
		return result;

	result *= weights[entry] * pdf;
	pdf *= weights[entry];

	EMeasure measure = BSDF::getMeasure(bRec.sampledType);
	for (size_t i=0; i<2; ++i) {
		if (entry == i)
			continue;
		pdf += bsdfs[i].pdf(bRec, measure) * weights[i];
		result += bsdfs[i].f(bRec, measure) * weights[i];
	}
	return result/pdf;
}

CUDA_FUNC_IN Spectrum FrDiel(float cosi, float cost, const Spectrum &etai,
                const Spectrum &etat) {
    Spectrum Rparl = ((etat * cosi) - (etai * cost)) /
                     ((etat * cosi) + (etai * cost));
    Spectrum Rperp = ((etai * cosi) - (etat * cost)) /
                     ((etai * cosi) + (etat * cost));
    return (Rparl*Rparl + Rperp*Rperp) / 2.f;
}

CUDA_FUNC_IN Spectrum Evaluate(float cosi, float eta_i, float eta_t)
{
    // Compute Fresnel reflectance for dielectric
    cosi = clamp(cosi, -1.f, 1.f);

    // Compute indices of refraction for dielectric
    bool entering = cosi > 0.;
    float ei = eta_i, et = eta_t;
    if (!entering)
        swapk(ei, et);

    // Compute _sint_ using Snell's law
    float sint = ei/et * sqrtf(MAX(0.f, 1.f - cosi*cosi));
    if (sint >= 1.) {
        // Handle total internal reflection
        return 1.;
    }
    else {
        float cost = sqrtf(MAX(0.f, 1.f - sint*sint));
        return FrDiel(fabsf(cosi), cost, Spectrum(ei), Spectrum(et));
    }
}


