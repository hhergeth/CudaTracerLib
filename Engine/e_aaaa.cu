#include "e_BSDF.h"

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

