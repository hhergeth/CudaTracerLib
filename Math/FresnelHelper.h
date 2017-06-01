#pragma once

#include "Vector.h"
#include "Spectrum.h"

namespace CudaTracerLib {

//Implementation of most methods copied from Mitsuba, some are PBRT material too.

class FresnelHelper
{
public:
	CUDA_FUNC_IN static float fresnelDielectric(float cosThetaI, float cosThetaT, float eta)
	{
		if (eta == 1)
			return 0.0f;

		float Rs = (cosThetaI - eta * cosThetaT)
			/ (cosThetaI + eta * cosThetaT);
		float Rp = (eta * cosThetaI - cosThetaT)
			/ (eta * cosThetaI + cosThetaT);

		/* No polarization -- return the unpolarized reflectance */
		return 0.5f * (Rs * Rs + Rp * Rp);
	}

	CUDA_FUNC_IN static float fresnelDielectricExt(float cosThetaI_, float &cosThetaT_, float eta)
	{
		if (eta == 1) {
			cosThetaT_ = -cosThetaI_;
			return 0.0f;
		}

		/* Using Snell's law, calculate the squared sine of the
		angle between the normal and the transmitted ray */
		float scale = (cosThetaI_ > 0) ? 1.0f / eta : eta,
			cosThetaTSqr = 1.0f - (1.0f - cosThetaI_*cosThetaI_) * (scale*scale);

		/* Check for total internal reflection */
		if (cosThetaTSqr <= 0.0f) {
			cosThetaT_ = 0.0f;
			return 1.0f;
		}

		/* Find the absolute cosines of the incident/transmitted rays */
		float cosThetaI = math::abs(cosThetaI_);
		float cosThetaT = math::safe_sqrt(cosThetaTSqr);

		float Rs = (cosThetaI - eta * cosThetaT)
			/ (cosThetaI + eta * cosThetaT);
		float Rp = (eta * cosThetaI - cosThetaT)
			/ (eta * cosThetaI + cosThetaT);

		cosThetaT_ = (cosThetaI_ > 0) ? -cosThetaT : cosThetaT;

		/* No polarization -- return the unpolarized reflectance */
		return 0.5f * (Rs * Rs + Rp * Rp);
	}

	CUDA_FUNC_IN static float fresnelConductorApprox(float cosThetaI, float eta, float k)
	{
		float cosThetaI2 = cosThetaI*cosThetaI;

		float tmp = (eta*eta + k*k) * cosThetaI2;

		float Rp2 = (tmp - (eta * (2 * cosThetaI)) + 1)
			/ (tmp + (eta * (2 * cosThetaI)) + 1);

		float tmpF = eta*eta + k*k;

		float Rs2 = (tmpF - (eta * (2 * cosThetaI)) + cosThetaI2) /
			(tmpF + (eta * (2 * cosThetaI)) + cosThetaI2);

		return 0.5f * (Rp2 + Rs2);
	}

	CUDA_FUNC_IN static Spectrum fresnelConductorApprox(float cosThetaI, const Spectrum &eta, const Spectrum &k)
	{
		float cosThetaI2 = cosThetaI*cosThetaI;

		Spectrum tmp = (eta*eta + k*k) * cosThetaI2;

		Spectrum Rp2 = (tmp - (eta * (2 * cosThetaI)) + Spectrum(1.0f))
			/ (tmp + (eta * (2 * cosThetaI)) + Spectrum(1.0f));

		Spectrum tmpF = eta*eta + k*k;

		Spectrum Rs2 = (tmpF - (eta * (2 * cosThetaI)) + Spectrum(cosThetaI2)) /
			(tmpF + (eta * (2 * cosThetaI)) + Spectrum(cosThetaI2));

		return 0.5f * (Rp2 + Rs2);
	}

	CUDA_FUNC_IN static float fresnelConductorExact(float cosThetaI, float eta, float k)
	{
		/* Modified from "Optics" by K.D. Moeller, University Science Books, 1988 */

		float cosThetaI2 = cosThetaI*cosThetaI,
			sinThetaI2 = 1 - cosThetaI2,
			sinThetaI4 = sinThetaI2*sinThetaI2;

		float temp1 = eta*eta - k*k - sinThetaI2,
			a2pb2 = math::sqrt(temp1*temp1 + 4 * k*k*eta*eta),
			a = math::sqrt(0.5f * (a2pb2 + temp1));

		float term1 = a2pb2 + cosThetaI2,
			term2 = 2 * a*cosThetaI;

		float Rs2 = (term1 - term2) / (term1 + term2);

		float term3 = a2pb2*cosThetaI2 + sinThetaI4,
			term4 = term2*sinThetaI2;

		float Rp2 = Rs2 * (term3 - term4) / (term3 + term4);

		return 0.5f * (Rp2 + Rs2);
	}

	CUDA_FUNC_IN static Spectrum fresnelConductorExact(float cosThetaI, const Spectrum &eta, const Spectrum &k)
	{
		/* Modified from "Optics" by K.D. Moeller, University Science Books, 1988 */

		float cosThetaI2 = cosThetaI*cosThetaI,
			sinThetaI2 = 1 - cosThetaI2,
			sinThetaI4 = sinThetaI2*sinThetaI2;

		Spectrum temp1 = eta*eta - k*k - Spectrum(sinThetaI2),
			a2pb2 = (temp1*temp1 + k*k*eta*eta * 4).safe_sqrt(),
			a = ((a2pb2 + temp1) * 0.5f).safe_sqrt();

		Spectrum term1 = a2pb2 + Spectrum(cosThetaI2),
			term2 = a*(2 * cosThetaI);

		Spectrum Rs2 = (term1 - term2) / (term1 + term2);

		Spectrum term3 = a2pb2*cosThetaI2 + Spectrum(sinThetaI4),
			term4 = term2*sinThetaI2;

		Spectrum Rp2 = Rs2 * (term3 - term4) / (term3 + term4);

		return 0.5f * (Rp2 + Rs2);
	}

	CUDA_FUNC_IN static NormalizedT<Vec3f> reflect(const NormalizedT<Vec3f> &wi, const NormalizedT<Vec3f> &n)
	{
		return (2 * dot(wi, n) * (n)-wi).normalized();
	}

	CUDA_FUNC_IN static Vec3f refract(const Vec3f &wi, const Vec3f &n, float eta, float cosThetaT)
	{
		if (cosThetaT < 0)
			eta = 1.0f / eta;

		return n * (dot(wi, n) * eta + cosThetaT) - wi * eta;
	}

	CUDA_FUNC_IN static Vec3f refract(const Vec3f &wi, const Vec3f &n, float eta)
	{
		if (eta == 1)
			return -1.0f * wi;

		float cosThetaI = dot(wi, n);
		if (cosThetaI > 0)
			eta = 1.0f / eta;

		/* Using Snell's law, calculate the squared sine of the
		angle between the normal and the transmitted ray */
		float cosThetaTSqr = 1.0f - (1.0f - cosThetaI*cosThetaI) * (eta*eta);

		/* Check for total internal reflection */
		if (cosThetaTSqr <= 0.0f)
			return Vec3f(0.0f);

		return n * (cosThetaI * eta - math::signum(cosThetaI) * math::sqrt(cosThetaTSqr)) - wi * eta;
	}

	CUDA_FUNC_IN static Vec3f refract(const Vec3f &wi, const Vec3f &n, float eta, float &cosThetaT, float &F)
	{
		float cosThetaI = dot(wi, n);
		F = fresnelDielectricExt(cosThetaI, cosThetaT, eta);

		if (F == 1.0f) /* Total internal reflection */
			return Vec3f(0.0f);

		if (cosThetaT < 0)
			eta = 1 / eta;

		return n * (eta * cosThetaI + cosThetaT) - wi * eta;
	}

	CUDA_FUNC_IN static float fresnelDielectricExt(float cosThetaI, float eta)
	{
		float cosThetaT;
		return fresnelDielectricExt(cosThetaI, cosThetaT, eta);
	}

	CTL_EXPORT CUDA_DEVICE CUDA_HOST static float fresnelDiffuseReflectance(float eta, bool fast);
};

}