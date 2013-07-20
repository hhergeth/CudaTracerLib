#pragma once

//#define FAST_BRDF

#include "..\Math\vector.h"

#define BXDF_SMALL 60
#define BXDF_LARGE 80

struct BSDFSample
{
   // BSDFSample Public Methods
   CUDA_FUNC_IN BSDFSample(float up0, float up1, float ucomp)
   {
       uDir[0] = up0;
       uDir[1] = up1;
       uComponent = ucomp;
   }
   CUDA_FUNC_IN BSDFSample(CudaRNG& rng)
   {
      uDir[0] = rng.randomFloat();
      uDir[1] = rng.randomFloat();
      uComponent = rng.randomFloat();
   }
   CUDA_FUNC_IN BSDFSample() { }
   float uDir[2], uComponent;
};

struct FresnelNoOp
{
	CUDA_FUNC_IN float3 Evaluate(float) const
	{
		return make_float3(1);
	}
};

struct FresnelConductor
{
	float3 eta;
	float3 k;
	CUDA_FUNC_IN float3 Evaluate(float cosi) const
	{
		return FrCond(fabsf(cosi), eta, k);
	}
};

struct FresnelDielectric
{
	float etai;
	float etat;
	CUDA_FUNC_IN float3 Evaluate(float cosi) const
	{
		cosi = clamp(cosi, -1.f, 1.f);
		bool entering = cosi > 0.;
		float ei = etai, et = etat;
		if (!entering)
		{
			float q = ei;
			ei = et;
			et = q;
		}
		float sint = ei/et * sqrtf(MAX(0.f, 1.f - cosi * cosi));
		float3 eval;
		if (sint >= 1.)
			eval = make_float3(1.0f);
		else
		{
			float cost = sqrtf(MAX(0.f, 1.f - sint*sint));
			eval = FrDiel(fabsf(cosi), cost, make_float3(ei), make_float3(et));
		}
		return eval;
	}
};

struct e_KernelFresnel
{
	unsigned int m_uType;
	union
	{
		FresnelNoOp NoOp;
		FresnelConductor Conductor;
		FresnelDielectric Dielectric;
	};
	CUDA_FUNC_IN e_KernelFresnel()
	{
		m_uType = 1;
	}
	CUDA_FUNC_IN e_KernelFresnel(const float3& eta, const float3& k)
	{
		m_uType = 2;
		Conductor.eta = eta;
		Conductor.k = k;
	}
	CUDA_FUNC_IN e_KernelFresnel(float etai, float etat)
	{
		m_uType = 3;
		Dielectric.etai = etai;
		Dielectric.etat = etat;
	}
	CUDA_FUNC_IN float3 Evaluate(float cosi) const
	{
		if(m_uType == 1)
			return NoOp.Evaluate(cosi);
		else if(m_uType == 2)
			return Conductor.Evaluate(cosi);
		else if(m_uType == 3)
			return Dielectric.Evaluate(cosi);
	}
};

enum BxDFType
{
    BSDF_REFLECTION   = 1,
    BSDF_TRANSMISSION = 2,
    BSDF_DIFFUSE      = 4,
    BSDF_GLOSSY       = 8,
    BSDF_SPECULAR     = 16,
    BSDF_ALL_TYPES        = BSDF_DIFFUSE |
                            BSDF_GLOSSY |
                            BSDF_SPECULAR,
    BSDF_ALL_REFLECTION   = BSDF_REFLECTION |
                            BSDF_ALL_TYPES,
    BSDF_ALL_TRANSMISSION = BSDF_TRANSMISSION |
                            BSDF_ALL_TYPES,
    BSDF_ALL              = BSDF_ALL_REFLECTION |
                            BSDF_ALL_TRANSMISSION
};

#define STD_Sample_f \
	CUDA_FUNC_IN float3 Sample_f(const float3& wo, float3* wi, float u1, float u2, float* pdf) const \
	{ \
		*wi = CosineSampleHemisphere(u1, u2); \
		if (wo.z < 0.) \
			wi->z *= -1.f; \
		*pdf = Pdf(wo, *wi); \
		return f(wo, *wi); \
	}

#define STD_Pdf \
	CUDA_FUNC_IN float Pdf(const float3& wo, const float3& wi) const \
	{ \
		return SameHemisphere(wo, wi) ? AbsCosTheta(wi) * INV_PI : 0.f; \
	}

#define STD_rho \
	CUDA_FUNC_IN float3 rho(const float3 &wo, int nSamples, const float *samples) const \
	{ \
		float3 r = make_float3(0); \
		for (int i = 0; i < nSamples; ++i) \
		{ \
			float3 wi; \
			float pdf = 0.0f; \
			float3 f = Sample_f(wo, &wi, samples[2*i], samples[2*i+1], &pdf); \
			if (pdf > 0.0f) \
				r += f * AbsCosTheta(wi) / pdf; \
		} \
		return r / float(nSamples); \
	}

#define STD_rho2 \
	CUDA_FUNC_IN float3 rho(int nSamples, const float *samples1, const float *samples2) const \
	{ \
		float3 r = make_float3(0); \
		for (int i = 0; i < nSamples; ++i) \
		{ \
			float3 wo, wi; \
			wo = UniformSampleHemisphere(samples1[2*i], samples1[2*i+1]); \
			float pdf_o = INV_TWOPI, pdf_i = 0.0f; \
			float3 f = Sample_f(wo, &wi, samples2[2*i], samples2[2*i+1], &pdf_i); \
			if (pdf_i > 0.0f) \
				r += f * AbsCosTheta(wi) * AbsCosTheta(wo) / (pdf_o * pdf_i); \
		} \
		return r / (PI * nSamples); \
	}

/*
struct e_KernelBrdf_
{
	CUDA_FUNC_IN float3 f(const float3& wo, const float3& wi) const
	{

	}

	STD_Pdf

	STD_Sample_f

	STD_rho

	STD_rho2
};
*/

#define e_KernelBrdf_Lambertain_TYPE 1
struct e_KernelBrdf_Lambertain
{
	CUDA_FUNC_IN e_KernelBrdf_Lambertain()
	{
		type = (BxDFType)(BSDF_REFLECTION | BSDF_DIFFUSE);
	}

	CUDA_FUNC_IN e_KernelBrdf_Lambertain(const float3& r)
	{
		type = (BxDFType)(BSDF_REFLECTION | BSDF_DIFFUSE);
		R = r;
	}

	CUDA_FUNC_IN float3 f(const float3& wo, const float3& wi) const
	{
		return R * INV_PI;
	}

	STD_Pdf

	STD_Sample_f

	CUDA_FUNC_IN float3 rho(const float3 &wo, int nSamples, const float *samples) const
	{
		return R;
	}

	CUDA_FUNC_IN float3 rho(int nSamples, const float *samples1, const float *samples2) const
	{
		return R;
	}

	TYPE_FUNC(e_KernelBrdf_Lambertain)
public:
	BxDFType type;
	float3 R;
};

#define e_KernelBrdf_SpecularReflection_TYPE 2
struct e_KernelBrdf_SpecularReflection
{
	CUDA_FUNC_IN e_KernelBrdf_SpecularReflection()
	{
		type = (BxDFType)(BSDF_REFLECTION | BSDF_SPECULAR);
	}

	CUDA_FUNC_IN e_KernelBrdf_SpecularReflection(const float3& r, const e_KernelFresnel& f)
	{
		type = (BxDFType)(BSDF_REFLECTION | BSDF_SPECULAR);
		R = r;
		fresnel = f;
	}

	CUDA_FUNC_IN float3 f(const float3& wo, const float3& wi) const
	{
		return make_float3(0);
	}

	CUDA_FUNC_IN float3 Sample_f(const float3& wo, float3* wi, float u1, float u2, float* pdf) const
	{
		*wi = make_float3(-wo.x, -wo.y, wo.z);
		*pdf = 1.f;
		return fresnel.Evaluate(CosTheta(wo)) * R / AbsCosTheta(*wi);
	}

	CUDA_FUNC_IN float Pdf(const float3& wi, const float3& wo) const
	{
		return 0;
	}

	STD_rho

	STD_rho2

	TYPE_FUNC(e_KernelBrdf_SpecularReflection)
public:
	BxDFType type;
	float3 R;
	e_KernelFresnel fresnel;
};

#define e_KernelBrdf_SpecularTransmission_TYPE 3
struct e_KernelBrdf_SpecularTransmission
{
	CUDA_FUNC_IN e_KernelBrdf_SpecularTransmission()
	{
		type = (BxDFType)(BSDF_TRANSMISSION | BSDF_SPECULAR);
	}

	CUDA_FUNC_IN e_KernelBrdf_SpecularTransmission(const float3& t, float _etai, float _etat)
	{
		type = (BxDFType)(BSDF_TRANSMISSION | BSDF_SPECULAR);
		T = t;
		fresnel.etai = etai = _etai;
		fresnel.etat = etat = _etat;
	}

	CUDA_FUNC_IN float3 f(const float3& wo, const float3& wi) const
	{
		return make_float3(0);
	}

	CUDA_FUNC_IN float3 Sample_f(const float3& wo, float3* wi, float u1, float u2, float* pdf) const
	{
		bool entering = CosTheta(wo) > 0.;
		float ei = etai, et = etat;
		if (!entering)
		{
			float q = ei;
			ei = et;
			et = q;
		}

		// Compute transmitted ray direction
		float sini2 = SinTheta2(wo);
		float eta = ei / et;
		float sint2 = eta * eta * sini2;

		// Handle total internal reflection for transmission
		if (sint2 >= 1.)
			return make_float3(0);
		float cost = sqrtf(MAX(0.f, 1.f - sint2));
		cost = entering ? -cost : cost;
		float sintOverSini = eta;
		*wi = make_float3(sintOverSini * -wo.x, sintOverSini * -wo.y, cost);
		*pdf = 1.f;
		float3 F = fresnel.Evaluate(CosTheta(wo));
		return (make_float3(1.0f)-F) * T / AbsCosTheta(*wi);
	}

	CUDA_FUNC_IN float Pdf(const float3& wi, const float3& wo) const
	{
		return 0;
	}

	STD_rho

	STD_rho2

	TYPE_FUNC(e_KernelBrdf_SpecularTransmission)
public:
	BxDFType type;
	float3 T;
	float etai, etat;
	FresnelDielectric fresnel;
};

#define e_KernelBrdf_OrenNayar_TYPE 4
struct e_KernelBrdf_OrenNayar
{
	CUDA_FUNC_IN e_KernelBrdf_OrenNayar()
	{
		type = BxDFType(BSDF_REFLECTION | BSDF_DIFFUSE);
	}

	CUDA_FUNC_IN e_KernelBrdf_OrenNayar(const float3& r, float sig)
	{
		type = BxDFType(BSDF_REFLECTION | BSDF_DIFFUSE);
		R = r;
		float sigma = Radians(sig);
        float sigma2 = sigma*sigma;
        A = 1.f - (sigma2 / (2.f * (sigma2 + 0.33f)));
        B = 0.45f * sigma2 / (sigma2 + 0.09f);
	}

	CUDA_FUNC_IN float3 f(const float3& wo, const float3& wi) const
	{
		float sinthetai = SinTheta(wi);
		float sinthetao = SinTheta(wo);
		// Compute cosine term of Oren-Nayar model
		float maxcos = 0.f;
		if (sinthetai > 1e-4 && sinthetao > 1e-4)
		{
			float sinphii = SinPhi(wi), cosphii = CosPhi(wi);
			float sinphio = SinPhi(wo), cosphio = CosPhi(wo);
			float dcos = cosphii * cosphio + sinphii * sinphio;
			maxcos = MAX(0.0f, dcos);
		}

		// Compute sine and tangent terms of Oren-Nayar model
		float sinalpha, tanbeta;
		if (AbsCosTheta(wi) > AbsCosTheta(wo))
		{
			sinalpha = sinthetao;
			tanbeta = sinthetai / AbsCosTheta(wi);
		}
		else
		{
			sinalpha = sinthetai;
			tanbeta = sinthetao / AbsCosTheta(wo);
		}
		return R * INV_PI * (A + B * maxcos * sinalpha * tanbeta);
	}

	STD_Pdf

	STD_Sample_f

	STD_rho

	STD_rho2

	TYPE_FUNC(e_KernelBrdf_OrenNayar)
public:
	BxDFType type;
	float3 R;
    float A, B;
};

template<typename MIC_DIST> struct e_KernelBrdf_Microfacet
{
	CUDA_FUNC_IN e_KernelBrdf_Microfacet()
	{
		type = BxDFType(BSDF_REFLECTION | BSDF_GLOSSY);
	}

	CUDA_FUNC_IN e_KernelBrdf_Microfacet(const float3& r, const e_KernelFresnel& f, const MIC_DIST& d)
	{
		type = BxDFType(BSDF_REFLECTION | BSDF_GLOSSY);
		R = r;
		fresnel = f;
		distribution = d;
	}

	CUDA_FUNC_IN float3 f(const float3& wo, const float3& wi) const
	{
		float cosThetaO = AbsCosTheta(wo);
		float cosThetaI = AbsCosTheta(wi);
		if (cosThetaI == 0.f || cosThetaO == 0.f)
			return make_float3(0.0f);
		float3 wh = wi + wo;
		if (wh.x == 0. && wh.y == 0. && wh.z == 0.)
			return make_float3(0.0f);
		wh = normalize(wh);
		float cosThetaH = dot(wi, wh);
		float3 F = fresnel.Evaluate(cosThetaH);
		return R * distribution.D(wh) * G(wo, wi, wh) * F / (4.f * cosThetaI * cosThetaO);
	}

	CUDA_FUNC_IN float Pdf(const float3& wi, const float3& wo) const
	{
		if (!SameHemisphere(wo, wi))
			return 0.f;
		return distribution.Pdf(wo, wi);
	}

	CUDA_FUNC_IN float3 Sample_f(const float3& wo, float3* wi, float u1, float u2, float* pdf) const
	{
		distribution.Sample_f(wo, wi, u1, u2, pdf);
		if (!SameHemisphere(wo, *wi))
			return make_float3(0.0f);
		return f(wo, *wi);
	}

	STD_rho

	STD_rho2
public:
	BxDFType type;
	float3 R;
	e_KernelFresnel fresnel;
	MIC_DIST distribution;
private:
	CUDA_FUNC_IN float G(const float3 &wo, const float3 &wi, const float3 &wh) const
	{
        float NdotWh = AbsCosTheta(wh);
        float NdotWo = AbsCosTheta(wo);
        float NdotWi = AbsCosTheta(wi);
        float WOdotWh = AbsDot(wo, wh);
		return MIN(1.f, MIN((2.f * NdotWh * NdotWo / WOdotWh), (2.f * NdotWh * NdotWi / WOdotWh)));
    }
};

struct e_KernelBrdf_BlinnDistribution
{
	CUDA_FUNC_IN e_KernelBrdf_BlinnDistribution()
	{

	}

	CUDA_FUNC_IN e_KernelBrdf_BlinnDistribution(float e)
	{
		exponent = e;
	}

	CUDA_FUNC_IN float D(const float3 &wh) const
	{
        float costhetah = AbsCosTheta(wh);
        return (exponent+2) * INV_TWOPI * powf(costhetah, exponent);
    }

	CUDA_FUNC_IN float Pdf(const float3& wi, const float3& wo) const
	{
		float3 wh = normalize(wo + wi);
		float costheta = AbsCosTheta(wh);
		float blinn_pdf = ((exponent + 1.f) * powf(costheta, exponent)) / (2.f * PI * 4.f * dot(wo, wh));
		if (dot(wo, wh) <= 0.f)
			blinn_pdf = 0.f;
		return blinn_pdf;
	}

	CUDA_FUNC_IN float3 Sample_f(const float3& wo, float3* wi, float u1, float u2, float* pdf) const
	{
		float costheta = powf(u1, 1.f / (exponent+1));
		float sintheta = sqrtf(MAX(0.f, 1.f - costheta*costheta));
		float phi = u2 * 2.f * PI;
		float3 wh = SphericalDirection(sintheta, costheta, phi);
		if (!SameHemisphere(wo, wh))
			wh = -wh;
		*wi = -1.0f * wo + 2.f * dot(wo, wh) * wh;
		float blinn_pdf = ((exponent + 1.f) * powf(costheta, exponent)) / (2.f * PI * 4.f * dot(wo, wh));
		if (dot(wo, wh) <= 0.f)
			blinn_pdf = 0.f;
		*pdf = blinn_pdf;
	}
public:
	float exponent;
};

struct e_KernelBrdf_AnisotropicDistribution
{
	CUDA_FUNC_IN e_KernelBrdf_AnisotropicDistribution()
	{

	}

	CUDA_FUNC_IN e_KernelBrdf_AnisotropicDistribution(float x, float y)
	{
		ex = x;
		ey = y;
	}

	CUDA_FUNC_IN float D(const float3 &wh) const
	{
		float costhetah = AbsCosTheta(wh);
		float d = 1.f - costhetah * costhetah;
		if (d == 0.f)
			return 0.f;
		float e = (ex * wh.x * wh.x + ey * wh.y * wh.y) / d;
		return sqrtf((ex+2.f) * (ey+2.f)) * INV_TWOPI * powf(costhetah, e);
	}

	CUDA_FUNC_IN float Pdf(const float3& wi, const float3& wo) const
	{
		float3 wh = normalize(wo + wi);
		float costhetah = AbsCosTheta(wh);
		float ds = 1.f - costhetah * costhetah;
		float anisotropic_pdf = 0.0f;
		if (ds > 0.f && dot(wo, wh) > 0.f)
		{
			float e = (ex * wh.x * wh.x + ey * wh.y * wh.y) / ds;
			float d = sqrtf((ex+1.f) * (ey+1.f)) * INV_TWOPI * powf(costhetah, e);
			anisotropic_pdf = d / (4.f * dot(wo, wh));
		}
		return anisotropic_pdf;
	}

	CUDA_FUNC_IN float3 Sample_f(const float3& wo, float3* wi, float u1, float u2, float* pdf) const
	{
		float phi, costheta;
		if (u1 < .25f)
		{
			sampleFirstQuadrant(4.f * u1, u2, &phi, &costheta);
		}
		else if (u1 < .5f)
		{
			u1 = 4.f * (.5f - u1);
			sampleFirstQuadrant(u1, u2, &phi, &costheta);
			phi = PI - phi;
		}
		else if (u1 < .75f)
		{
			u1 = 4.f * (u1 - .5f);
			sampleFirstQuadrant(u1, u2, &phi, &costheta);
			phi += PI;
		}
		else
		{
			u1 = 4.f * (1.f - u1);
			sampleFirstQuadrant(u1, u2, &phi, &costheta);
			phi = 2.f * PI - phi;
		}
		float sintheta = sqrtf(MAX(0.f, 1.f - costheta*costheta));
		float3 wh = SphericalDirection(sintheta, costheta, phi);
		if (!SameHemisphere(wo, wh))
			wh = -wh;
		*wi = -1.0f * wo + 2.f * dot(wo, wh) * wh;
		float costhetah = AbsCosTheta(wh);
		float ds = 1.f - costhetah * costhetah;
		float anisotropic_pdf = 0.f;
		if (ds > 0.f && dot(wo, wh) > 0.f)
		{
			float e = (ex * wh.x * wh.x + ey * wh.y * wh.y) / ds;
			float d = sqrtf((ex+1.f) * (ey+1.f)) * INV_TWOPI * powf(costhetah, e);
			anisotropic_pdf = d / (4.f * dot(wo, wh));
		}
		*pdf = anisotropic_pdf;
	}
public:
	float ex, ey;
private:
	CUDA_FUNC_IN void sampleFirstQuadrant(float u1, float u2, float *phi, float *costheta) const
	{
		if (ex == ey)
			*phi = PI * u1 * 0.5f;
		else *phi = atanf(sqrtf((ex+1.f) / (ey+1.f)) * tanf(PI * u1 * 0.5f));
		float cosphi = cosf(*phi), sinphi = sinf(*phi);
		*costheta = powf(u2, 1.f/(ex * cosphi * cosphi + ey * sinphi * sinphi + 1));
	}
};

struct e_KernelBrdf_MicrofacetDistribution
{
private:
	unsigned char data[8];
	unsigned int type;
#define AS(T) ((T*)data)
public:
	CUDA_FUNC_IN e_KernelBrdf_MicrofacetDistribution()
	{
		type = 0;//yeah thats fucked, but idk
	}

	CUDA_FUNC_IN e_KernelBrdf_MicrofacetDistribution(const e_KernelBrdf_BlinnDistribution& d)
	{
		*AS(e_KernelBrdf_BlinnDistribution) = d;
		type = 0;
	}

	CUDA_FUNC_IN e_KernelBrdf_MicrofacetDistribution(const e_KernelBrdf_AnisotropicDistribution& d)
	{
		*AS(e_KernelBrdf_AnisotropicDistribution) = d;
		type = 1;
	}

	CUDA_FUNC_IN float D(const float3 &wh) const
	{
		if(type)
			return AS(e_KernelBrdf_BlinnDistribution)->D(wh);
		else return AS(e_KernelBrdf_AnisotropicDistribution)->D(wh);
    }

	CUDA_FUNC_IN float Pdf(const float3& wi, const float3& wo) const
	{
		if(type)
			return AS(e_KernelBrdf_BlinnDistribution)->Pdf(wi, wo);
		else return AS(e_KernelBrdf_AnisotropicDistribution)->Pdf(wi, wo);
	}

	CUDA_FUNC_IN float3 Sample_f(const float3& wo, float3* wi, float u1, float u2, float* pdf) const
	{
		if(type)
			return AS(e_KernelBrdf_BlinnDistribution)->Sample_f(wo, wi, u1, u2, pdf);
		else return AS(e_KernelBrdf_AnisotropicDistribution)->Sample_f(wo, wi, u1, u2, pdf);
	}
};

#define e_KernelBrdf_Blinn_TYPE 5
struct e_KernelBrdf_Blinn : public e_KernelBrdf_Microfacet<e_KernelBrdf_BlinnDistribution>
{
	CUDA_FUNC_IN e_KernelBrdf_Blinn(const float3& r, const e_KernelFresnel& f, const e_KernelBrdf_BlinnDistribution& d)
		: e_KernelBrdf_Microfacet<e_KernelBrdf_BlinnDistribution>(r, f, d)
	{
	}

	TYPE_FUNC(e_KernelBrdf_Blinn)
};

#define e_KernelBrdf_Anisotropic_TYPE 6
struct e_KernelBrdf_Anisotropic : public e_KernelBrdf_Microfacet<e_KernelBrdf_AnisotropicDistribution>
{
	CUDA_FUNC_IN e_KernelBrdf_Anisotropic(const float3& r, const e_KernelFresnel& f, const e_KernelBrdf_AnisotropicDistribution& d)
		: e_KernelBrdf_Microfacet<e_KernelBrdf_AnisotropicDistribution>(r, f, d)
	{
	}

	TYPE_FUNC(e_KernelBrdf_Blinn)
};

#define e_KernelBrdf_FresnelBlend_TYPE 7
struct e_KernelBrdf_FresnelBlend
{
	CUDA_FUNC_IN e_KernelBrdf_FresnelBlend()
	{
		type = BxDFType(BSDF_REFLECTION | BSDF_GLOSSY);
	}

	CUDA_FUNC_IN e_KernelBrdf_FresnelBlend(const float3& rd, const float3& rs, const e_KernelBrdf_MicrofacetDistribution& d)
	{
		type = BxDFType(BSDF_REFLECTION | BSDF_GLOSSY);
		Rd = rd;
		Rs = rs;
		distribution = d;
	}

	CUDA_FUNC_IN float3 f(const float3& wo, const float3& wi) const
	{
		float3 diffuse = (28.0f/(23.0f*PI)) * Rd * (make_float3(1.0f) - Rs) * (1.0f - powf(1.0f - 0.5f * AbsCosTheta(wi), 5.0f)) * (1.0f - powf(1.0f - 0.5f * AbsCosTheta(wo), 5.0f));
		float3 wh = wi + wo;
		if (wh.x == 0.0f && wh.y == 0.0f && wh.z == 0.0f)
			return make_float3(0.0f);
		wh = normalize(wh);
		float3 specular = distribution.D(wh) / (4.f * AbsDot(wi, wh) * MAX(AbsCosTheta(wi), AbsCosTheta(wo))) *	SchlickFresnel(dot(wi, wh));
		return diffuse + specular;
	}

	STD_Pdf

	STD_Sample_f

	STD_rho

	STD_rho2

	TYPE_FUNC(e_KernelBrdf_FresnelBlend)
public:
	BxDFType type;
	float3 Rd;
	float3 Rs;
	e_KernelBrdf_MicrofacetDistribution distribution;
private:
	CUDA_FUNC_IN float3 SchlickFresnel(float costheta) const
	{
        return Rs + powf(1 - costheta, 5.f) * (make_float3(1.) - Rs);
    }
};

template<int BUFFER_SIZE> struct e_KernelBXDF
{
#define CALL_TYPE(t,f,r) \
	case t##_TYPE : \
		r ((t*)Data)->f; \
		break;
#define CALL_FUNC(r,f) \
	switch (m_uType) \
	{ \
		CALL_TYPE(e_KernelBrdf_Lambertain, f, r) \
		CALL_TYPE(e_KernelBrdf_SpecularReflection, f, r) \
		CALL_TYPE(e_KernelBrdf_SpecularTransmission, f, r) \
		CALL_TYPE(e_KernelBrdf_OrenNayar, f, r) \
		CALL_TYPE(e_KernelBrdf_Blinn, f, r) \
		CALL_TYPE(e_KernelBrdf_Anisotropic, f, r) \
		CALL_TYPE(e_KernelBrdf_FresnelBlend, f, r) \
	}
private:
	unsigned char Data[BUFFER_SIZE];
	int		m_uType;
public:
	CUDA_FUNC_IN e_KernelBXDF()
	{
		m_uType = 0;
	}

	CUDA_FUNC_IN e_KernelBXDF(unsigned int a_Type)
	{
		m_uType = a_Type;
	}

	template<typename T> CUDA_FUNC_IN void SetData(T& val)
	{
		m_uType = T::TYPE();
		*(T*)Data = val;
	}

	CUDA_FUNC_IN BxDFType getType() const
	{
		return *(BxDFType*)Data;
	}

	CUDA_FUNC_IN float3 f(const float3& wo, const float3& wi) const
	{
		CALL_FUNC(return, f(wo, wi))
	}

	CUDA_FUNC_IN float3 Sample_f(const float3& wo, float3* wi, float u1, float u2, float* pdf) const
	{
		CALL_FUNC(return, Sample_f(wo, wi, u1, u2, pdf));
	}

	CUDA_FUNC_IN float Pdf(const float3& wi, const float3& wo) const
	{
		CALL_FUNC(return, Pdf(wi, wo));
	}

	CUDA_FUNC_IN bool MatchesFlags(BxDFType flags) const
	{
		BxDFType type = getType();
		return (type & flags) == type;
	}

	CUDA_FUNC_IN float3 rho(const float3 &wo, int nSamples, const float *samples) const
	{
		CALL_FUNC(return, rho(wo, nSamples, samples));
	}

	CUDA_FUNC_IN float3 rho(int nSamples, const float *samples1, const float *samples2) const
	{
		CALL_FUNC(return, rho(nSamples, samples1, samples2));
	}
#undef CALL_TYPE
#undef CALL_FUNC
};

#ifndef FAST_BRDF
#define BXDF_NUM_Brdf 4
struct e_KernelBSDF
{
private:
	e_KernelBXDF<BXDF_LARGE> m_sBXDF[BXDF_NUM_Brdf];
	unsigned int m_uNumUsed;
public:
	float3 ng;
	Onb sys;
	float Eta;
public:
	CUDA_FUNC_IN e_KernelBSDF()
		: Eta(1)
	{
	}

	CUDA_FUNC_IN e_KernelBSDF(const Onb& _sys, const float3& _ng)
		: sys(_sys), ng(_ng), Eta(1)
	{
		m_uNumUsed = 0;
	}

	CUDA_FUNC_IN float Pdf(const float3& woW, const float3& wiW, BxDFType flags = BSDF_ALL) const
	{
		if (m_uNumUsed == 0)
			return 0.0f;
		float3 wo = WorldToLocal(woW), wi = WorldToLocal(wiW);
		float pdf = 0.f;
		int matchingComps = 0;
		for (int i = 0; i < m_uNumUsed; ++i)
			if (m_sBXDF[i].MatchesFlags(flags))
			{
				++matchingComps;
				pdf += m_sBXDF[i].Pdf(wo, wi);
			}
		float v = matchingComps > 0 ? pdf / matchingComps : 0.f;
		return v;
	}

	CUDA_FUNC_IN float3 Sample_f(const float3& woW, float3* wiW, const BSDFSample& sample, float* pdf, BxDFType flags = BSDF_ALL, BxDFType *sampledType = NULL) const
	{
		int matchingComps = NumComponents(flags);
		if (matchingComps == 0)
		{
			*pdf = 0.f;
			if (sampledType)
				*sampledType = BxDFType(0);
			return make_float3(0.f);
		}
		int which = MIN(Floor2Int(sample.uComponent * float(matchingComps)), matchingComps-1), count = which;

		unsigned int bxdfIndex = -1;
		const e_KernelBXDF<BXDF_LARGE>* bxdf = 0;
		for (int i = 0; i < m_uNumUsed; ++i)
			if (m_sBXDF[i].MatchesFlags(flags) && count-- == 0)
			{
				bxdfIndex = i;
				bxdf = m_sBXDF + i;
				break;
			}

		float3 wo = WorldToLocal(woW);
		float3 wi;
		*pdf = 0.0f;
		float3 F = bxdf->Sample_f(wo, &wi, sample.uDir[0], sample.uDir[1], pdf);

		if (*pdf == 0.0f)
		{
			if (sampledType)
				*sampledType = BxDFType(0);
			return make_float3(0.0f);
		}
		if (sampledType)
			*sampledType = bxdf->getType();
		*wiW = LocalToWorld(wi);
		
		if (!(bxdf->getType() & BSDF_SPECULAR) && matchingComps > 1)
			for (int i = 0; i < m_uNumUsed; ++i)
				if (i != bxdfIndex && m_sBXDF[i].MatchesFlags(flags))
					*pdf += m_sBXDF[i].Pdf(wo, wi);
		if (matchingComps > 1)
			*pdf /= matchingComps;

		// Compute value of BSDF for sampled direction
		if (!(bxdf->getType() & BSDF_SPECULAR))
		{
			F = make_float3(0);
			if (dot(*wiW, ng) * dot(woW, ng) > 0) // ignore BTDFs
				flags = BxDFType(flags & ~BSDF_TRANSMISSION);
			else // ignore BRDFs
				flags = BxDFType(flags & ~BSDF_REFLECTION);
			for (int i = 0; i < m_uNumUsed; ++i)
				if (m_sBXDF[i].MatchesFlags(flags))
					F += m_sBXDF[i].f(wo, wi);
		}
		return F;
	}

	CUDA_FUNC_IN float3 f(const float3& woW, const float3& wiW, BxDFType flags = BSDF_ALL) const
	{
		float3 wi = WorldToLocal(wiW), wo = WorldToLocal(woW);
		if (dot(wiW, ng) * dot(woW, ng) > 0) // ignore BTDFs
			flags = BxDFType(flags & ~BSDF_TRANSMISSION);
		else // ignore BRDFs
			flags = BxDFType(flags & ~BSDF_REFLECTION);
		float3 ret = make_float3(0.0f);
		for (int i = 0; i < m_uNumUsed; ++i)
			if (m_sBXDF[i].MatchesFlags(flags))
				ret += m_sBXDF[i].f(wo, wi);
		return ret;
	}
	
	CUDA_FUNC_IN float3 rho(CudaRNG& rng, unsigned char* a_TmpBuffer, BxDFType flags = BSDF_ALL, int sqrtSamples = 6) const
	{
		int nSamples = sqrtSamples * sqrtSamples;
		float *s1 = (float*)a_TmpBuffer;
		StratifiedSample2D(s1, sqrtSamples, sqrtSamples, rng);
		float *s2 = (float*)a_TmpBuffer + nSamples * 2;
		StratifiedSample2D(s2, sqrtSamples, sqrtSamples, rng);

		float3 ret = make_float3(0);
		for (int i = 0; i < m_uNumUsed; ++i)
			if (m_sBXDF[i].MatchesFlags(flags))
				ret += m_sBXDF[i].rho(nSamples, s1, s2);
		return ret;
	}
	
	CUDA_FUNC_IN float3 rho(const float3& wo, CudaRNG& rng, unsigned char* a_TmpBuffer, BxDFType flags = BSDF_ALL, int sqrtSamples = 6) const
	{
		int nSamples = sqrtSamples * sqrtSamples;
		float *s1 = (float*)a_TmpBuffer;
		StratifiedSample2D(s1, sqrtSamples, sqrtSamples, rng);
		float3 ret = make_float3(0);
		for (int i = 0; i < m_uNumUsed; ++i)
			if (m_sBXDF[i].MatchesFlags(flags))
				ret += m_sBXDF[i].rho(wo, nSamples, s1);
		return ret;
	}

	CUDA_FUNC_IN float3 IntegratePdf(float3& f, float pdf, float3& wi)
	{
		return f * AbsDot(wi, sys.m_normal) / pdf;
	}

	CUDA_FUNC_IN unsigned int NumComponents() const
	{
		return m_uNumUsed;
	}

	CUDA_FUNC_IN unsigned int NumComponents(BxDFType flags) const
	{
		unsigned int r = 0;
		for(int i = 0; i < m_uNumUsed; i++)
			if(m_sBXDF[i].MatchesFlags(flags))
				r++;
		return r;
	}

	template<typename T> CUDA_FUNC_IN void Add(T& val)
	{
		m_sBXDF[m_uNumUsed++].SetData(val);
	}

	CUDA_FUNC_IN float3 WorldToLocal(const float3& v) const
	{
		return sys.worldTolocal(v);
	}

	CUDA_FUNC_IN float3 LocalToWorld(const float3& v) const
	{
		return sys.localToworld(v);
	}
};
#else
struct e_KernelBSDF
{
public:
	const float3 ng;
	const Onb sys;
	const float Eta;
	CUDA_FUNC_IN e_KernelBSDF(const Onb& _sys, const float3& _ng)
		: sys(_sys), ng(_ng), Eta(1)
	{
	}

	CUDA_FUNC_IN float Pdf(const float3& woW, const float3& wiW, BxDFType flags = BSDF_ALL) const
	{
		return dot(woW, wiW) > 0.0f ? AbsDot(wiW, ng) * INV_PI : 0.f;
	}

	CUDA_FUNC_IN float3 Sample_f(const float3& woW, float3* wiW, const BSDFSample& sample, float* pdf, BxDFType flags = BSDF_ALL, BxDFType *sampledType = NULL) const
	{
		*wiW = SampleCosineHemisphere(ng, sample.uDir[0], sample.uDir[1]);
		*pdf = AbsDot(*wiW, ng) * INV_PI;
		if(sampledType)
			*sampledType = BSDF_DIFFUSE;
		return make_float3(0.75f) * INV_PI;
	}

	CUDA_FUNC_IN float3 rho(CudaRNG& rng, unsigned char* a_TmpBuffer, BxDFType flags = BSDF_ALL, int sqrtSamples = 6) const
	{
		return make_float3(0.75f);
	}

	CUDA_FUNC_IN float3 rho(const float3& wo, CudaRNG& rng, unsigned char* a_TmpBuffer, BxDFType flags = BSDF_ALL, int sqrtSamples = 6) const
	{
		return make_float3(0.75f);
	}

	CUDA_FUNC_IN float3 IntegratePdf(float3& f, float pdf, float3& wi)
	{
		return f * AbsDot(wi, sys.m_normal) / pdf;
	}

	CUDA_FUNC_IN unsigned int NumComponents() const
	{
		return 1;
	}

	CUDA_FUNC_IN unsigned int NumComponents(BxDFType flags) const
	{
		return (flags & BSDF_DIFFUSE) == BSDF_DIFFUSE;
	}

	CUDA_FUNC_IN void Add(e_KernelBXDF<BXDF_LARGE>& b)
	{
	}

	CUDA_FUNC_IN float3 WorldToLocal(const float3& v) const
	{
		return sys.worldTolocal(v);
	}

	CUDA_FUNC_IN float3 LocalToWorld(const float3& v) const
	{
		return sys.localToworld(v);
	}
};
#endif

struct e_KernelBSSRDF
{
	CUDA_FUNC_IN e_KernelBSSRDF()
	{
	}

	CUDA_FUNC_IN e_KernelBSSRDF(float _e, float3& sa, float3& sps)
	{
		e = _e;
		sig_a = sa;
		sigp_s = sps;
	}

    float e;
    float3 sig_a, sigp_s;
};