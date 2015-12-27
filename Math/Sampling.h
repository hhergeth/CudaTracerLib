#pragma once

#include "Vector.h"
#include "Spectrum.h"

namespace CudaTracerLib {

struct CudaRNG;

//Implementation of most methods copied from Mitsuba, some are PBRT material too.

class MonteCarlo
{
public:
	CUDA_FUNC_IN static bool Quadratic(float A, float B, float C, float *t0, float *t1)
	{
		// Find quadratic discriminant
		float discrim = B * B - 4.f * A * C;
		if (discrim <= 0.) return false;
		float rootDiscrim = math::sqrt(discrim);

		// Compute quadratic _t_ values
		float q;
		if (B < 0) q = -.5f * (B - rootDiscrim);
		else       q = -.5f * (B + rootDiscrim);
		*t0 = q / A;
		*t1 = C / q;
		if (*t0 > *t1)
			swapk(*t0, *t1);
		return true;
	}
	CUDA_DEVICE CUDA_HOST static void RejectionSampleDisk(float *x, float *y, CudaRNG &rng);
	CUDA_FUNC_IN static Vec3f UniformSampleHemisphere(float u1, float u2)
	{
		float z = u1;
		float r = math::sqrt(max(0.f, 1.f - z*z));
		float phi = 2 * PI * u2;
		float x = r * cosf(phi);
		float y = r * sinf(phi);
		return Vec3f(x, y, z);
	}
	CUDA_FUNC_IN static float  UniformHemispherePdf()
	{
		return 1.0f / (2.0f * PI);
	}
	CUDA_FUNC_IN static Vec3f UniformSampleSphere(float u1, float u2)
	{
		float z = 1.f - 2.f * u1;
		float r = math::sqrt(max(0.f, 1.f - z*z));
		float phi = 2.f * PI * u2;
		float x = r * cosf(phi);
		float y = r * sinf(phi);
		return Vec3f(x, y, z);
	}
	CUDA_FUNC_IN static float  UniformSpherePdf()
	{
		return 1.f / (4.f * PI);
	}
	CUDA_FUNC_IN static Vec3f UniformSampleCone(float u1, float u2, float costhetamax)
	{
		float costheta = (1.f - u1) + u1 * costhetamax;
		float sintheta = math::sqrt(1.f - costheta*costheta);
		float phi = u2 * 2.f * PI;
		return Vec3f(cosf(phi) * sintheta, sinf(phi) * sintheta, costheta);
	}
	CUDA_FUNC_IN static Vec3f UniformSampleCone(float u1, float u2, float costhetamax, const Vec3f &x, const Vec3f &y, const Vec3f &z)
	{
		float costheta = math::lerp(costhetamax, 1.f, u1);
		float sintheta = math::sqrt(1.f - costheta*costheta);
		float phi = u2 * 2.f * PI;
		return cosf(phi) * sintheta * x + sinf(phi) * sintheta * y + costheta * z;
	}
	CUDA_FUNC_IN static float  UniformConePdf(float cosThetaMax)
	{
		return 1.f / (2.f * PI * (1.f - cosThetaMax));
	}
	CUDA_FUNC_IN static void UniformSampleDisk(float u1, float u2, float *x, float *y)
	{
		float r = math::sqrt(u1);
		float theta = 2.0f * PI * u2;
		*x = r * cosf(theta);
		*y = r * sinf(theta);
	}
	CUDA_FUNC_IN static void ConcentricSampleDisk(float u1, float u2, float *dx, float *dy)
	{
		float r, theta;
		// Map uniform random numbers to $[-1,1]^2$
		float sx = 2 * u1 - 1;
		float sy = 2 * u2 - 1;

		// Map square to $(r,\theta)$

		// Handle degeneracy at the origin
		if (sx == 0.0 && sy == 0.0) {
			*dx = 0.0;
			*dy = 0.0;
			return;
		}
		if (sx >= -sy) {
			if (sx > sy) {
				// Handle first region of disk
				r = sx;
				if (sy > 0.0) theta = sy / r;
				else          theta = 8.0f + sy / r;
			}
			else {
				// Handle second region of disk
				r = sy;
				theta = 2.0f - sx / r;
			}
		}
		else {
			if (sx <= sy) {
				// Handle third region of disk
				r = -sx;
				theta = 4.0f - sy / r;
			}
			else {
				// Handle fourth region of disk
				r = -sy;
				theta = 6.0f + sx / r;
			}
		}
		theta *= PI / 4.f;
		*dx = r * cosf(theta);
		*dy = r * sinf(theta);
	}
	CUDA_FUNC_IN static Vec3f CosineSampleHemisphere(float u1, float u2) {
		Vec3f ret;
		ConcentricSampleDisk(u1, u2, &ret.x, &ret.y);
		ret.z = sqrt(max(0.f, 1.f - ret.x*ret.x - ret.y*ret.y));
		return ret;
	}
	CUDA_FUNC_IN static float CosineHemispherePdf(float costheta, float phi)
	{
		return costheta / PI;
	}
	CUDA_DEVICE CUDA_HOST static void StratifiedSample1D(float *samples, int nSamples, CudaRNG &rng, bool jitter = true);
	CUDA_DEVICE CUDA_HOST static void StratifiedSample2D(float *samples, int nx, int ny, CudaRNG &rng, bool jitter = true);

	CUDA_FUNC_IN static float SphericalTheta(const Vec3f &v)
	{
		return acosf(math::clamp(v.z, -1.f, 1.f));
	}

	CUDA_FUNC_IN static float SphericalPhi(const Vec3f &v)
	{
		float p = atan2f(v.y, v.x);
		return (p < 0.f) ? p + 2.f * PI : p;
	}

	CUDA_FUNC_IN static float BalanceHeuristic(int nf, float fPdf, int ng, float gPdf)
	{
		return (nf * fPdf) / (nf * fPdf + ng * gPdf);
	}

	CUDA_FUNC_IN static float PowerHeuristic(int nf, float fPdf, int ng, float gPdf)
	{
		float f = nf * fPdf, g = ng * gPdf;
		return (f*f) / (f*f + g*g);
	}

	CUDA_FUNC_IN static Vec3f SphericalDirection(float theta, float phi)
	{
		float sinTheta, cosTheta, sinPhi, cosPhi;

		sincos(theta, &sinTheta, &cosTheta);
		sincos(phi, &sinPhi, &cosPhi);

		return Vec3f(
			sinTheta * cosPhi,
			sinTheta * sinPhi,
			cosTheta
			);
	}

	CUDA_FUNC_IN static Vec2f toSphericalCoordinates(const Vec3f &v)
	{
		Vec2f result = Vec2f(
			acos(v.z),
			atan2(v.y, v.x)
			);
		if (result.y < 0)
			result.y += 2 * PI;
		return result;
	}

	CUDA_FUNC_IN static bool solveLinearSystem2x2(const float a[2][2], const float b[2], float x[2])
	{
		float det = a[0][0] * a[1][1] - a[0][1] * a[1][0];

		if (math::abs(det) <= RCPOVERFLOW)
			return false;

		float inverse = (float) 1.0f / det;

		x[0] = (a[1][1] * b[0] - a[0][1] * b[1]) * inverse;
		x[1] = (a[0][0] * b[1] - a[1][0] * b[0]) * inverse;

		return true;
	}

	CUDA_DEVICE CUDA_HOST static void stratifiedSample1D(CudaRNG& random, float *dest, int count, bool jitter);

	CUDA_DEVICE CUDA_HOST static void stratifiedSample2D(CudaRNG& random, Vec2f *dest, int countX, int countY, bool jitter);

	CUDA_DEVICE CUDA_HOST static void latinHypercube(CudaRNG& random, float *dest, unsigned int nSamples, size_t nDim);

	CUDA_FUNC_IN static float fresnelDielectric(float cosThetaI, float cosThetaT, float eta) {
		if (eta == 1)
			return 0.0f;

		float Rs = (cosThetaI - eta * cosThetaT)
			/ (cosThetaI + eta * cosThetaT);
		float Rp = (eta * cosThetaI - cosThetaT)
			/ (eta * cosThetaI + cosThetaT);

		/* No polarization -- return the unpolarized reflectance */
		return 0.5f * (Rs * Rs + Rp * Rp);
	}

	CUDA_FUNC_IN static float fresnelDielectricExt(float cosThetaI_, float &cosThetaT_, float eta) {
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

	CUDA_FUNC_IN static float fresnelConductorApprox(float cosThetaI, float eta, float k) {
		float cosThetaI2 = cosThetaI*cosThetaI;

		float tmp = (eta*eta + k*k) * cosThetaI2;

		float Rp2 = (tmp - (eta * (2 * cosThetaI)) + 1)
			/ (tmp + (eta * (2 * cosThetaI)) + 1);

		float tmpF = eta*eta + k*k;

		float Rs2 = (tmpF - (eta * (2 * cosThetaI)) + cosThetaI2) /
			(tmpF + (eta * (2 * cosThetaI)) + cosThetaI2);

		return 0.5f * (Rp2 + Rs2);
	}

	CUDA_FUNC_IN static Spectrum fresnelConductorApprox(float cosThetaI, const Spectrum &eta, const Spectrum &k) {
		float cosThetaI2 = cosThetaI*cosThetaI;

		Spectrum tmp = (eta*eta + k*k) * cosThetaI2;

		Spectrum Rp2 = (tmp - (eta * (2 * cosThetaI)) + Spectrum(1.0f))
			/ (tmp + (eta * (2 * cosThetaI)) + Spectrum(1.0f));

		Spectrum tmpF = eta*eta + k*k;

		Spectrum Rs2 = (tmpF - (eta * (2 * cosThetaI)) + Spectrum(cosThetaI2)) /
			(tmpF + (eta * (2 * cosThetaI)) + Spectrum(cosThetaI2));

		return 0.5f * (Rp2 + Rs2);
	}

	CUDA_FUNC_IN static float fresnelConductorExact(float cosThetaI, float eta, float k) {
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

	CUDA_FUNC_IN static Spectrum fresnelConductorExact(float cosThetaI, const Spectrum &eta, const Spectrum &k) {
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

	CUDA_FUNC_IN static Vec3f reflect(const Vec3f &wi, const Vec3f &n) {
		return 2 * dot(wi, n) * (n)-wi;
	}

	CUDA_FUNC_IN static Vec3f refract(const Vec3f &wi, const Vec3f &n, float eta, float cosThetaT) {
		if (cosThetaT < 0)
			eta = 1.0f / eta;

		return n * (dot(wi, n) * eta + cosThetaT) - wi * eta;
	}

	CUDA_FUNC_IN static Vec3f refract(const Vec3f &wi, const Vec3f &n, float eta) {
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

	CUDA_FUNC_IN static Vec3f refract(const Vec3f &wi, const Vec3f &n, float eta, float &cosThetaT, float &F) {
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
		return MonteCarlo::fresnelDielectricExt(cosThetaI, cosThetaT, eta);
	}

	CUDA_DEVICE CUDA_HOST static unsigned int sampleReuse(float *cdf, unsigned int size, float &sample, float& pdf);

	//http://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
	// point p with respect to triangle (a, b, c)
	CUDA_FUNC_IN static bool Barycentric(const Vec3f& p, const Vec3f& a, const Vec3f& b, const Vec3f& c, float& u, float& v)
	{
		Vec3f v0 = b - a, v1 = c - a, v2 = p - a;
		float d00 = dot(v0, v0);
		float d01 = dot(v0, v1);
		float d11 = dot(v1, v1);
		float d20 = dot(v2, v0);
		float d21 = dot(v2, v1);
		float denom = d00 * d11 - d01 * d01;
		v = (d11 * d20 - d01 * d21) / denom;
		float w = (d00 * d21 - d01 * d20) / denom;
		u = 1.0f - v - w;
		return 0 <= v && v <= 1 && 0 <= u && u <= 1 && 0 <= w && w <= 1;
	}
};

}