#pragma once

#include "cutil_math.h"
#include "..\Base\CudaRandom.h"
#include "..\Base\STL.h"

template<int N> struct Distribution1D
{
	Distribution1D()
	{

	}
    Distribution1D(const float *f, int n)
	{
		if(n > N)
			throw 1;
        count = n;
        memcpy(func, f, n*sizeof(float));
        cdf[0] = 0.0f;
        for (int i = 1; i < count+1; ++i)
            cdf[i] = cdf[i-1] + func[i-1] / float(n);
        funcInt = cdf[count];
        if (funcInt == 0.f)
            for (int i = 1; i < n+1; ++i)
                cdf[i] = float(i) / float(n);
        else for (int i = 1; i < n+1; ++i)
                cdf[i] /= funcInt;
    }
    CUDA_FUNC_IN float SampleContinuous(float u, float *pdf, int *off = NULL) const
	{
        const float *ptr = STL_upper_bound(cdf, cdf+count+1, u);
        int offset = MAX(0, int(ptr-cdf-1));
        if (off)
			*off = offset;

        // Compute offset along CDF segment
        float du = (u - cdf[offset]) / (cdf[offset+1] - cdf[offset]);

        // Compute PDF for sampled offset
        if (pdf) *pdf = func[offset] / funcInt;

        // Return $x\in{}[0,1)$ corresponding to sample
        return (offset + du) / count;
    }
    CUDA_FUNC_IN int SampleDiscrete(float u, float *pdf) const
	{
        const float *ptr = STL_upper_bound(cdf, cdf+count+1, u);
		int offset = MAX(0, int(ptr-cdf-1));
        if (pdf) *pdf = func[offset] / (funcInt * count);
        return offset;
    }
public:
    float func[N];
	float cdf[N];
    float funcInt;
    int count;
};

template<int NU, int NV> struct Distribution2D
{
	Distribution2D()
	{

	}

    Distribution2D(const float *data, unsigned int nu, unsigned int nv)
	{
		this->nu = nu;
		this->nv = nv;
		for (int v = 0; v < nv; ++v)
			pConditionalV[v] = Distribution1D<NU>(&data[v*nu], nu);
		float marginalFunc[NV];
		for (int v = 0; v < nv; ++v)
			marginalFunc[v] = pConditionalV[v].funcInt;
		pMarginal = Distribution1D<NV>(&marginalFunc[0], nv);
	}

	void Initialize(const float *data, unsigned int nu, unsigned int nv)
	{
		this->nu = nu;
		this->nv = nv;
		for (unsigned int v = 0; v < nv; ++v)
			pConditionalV[v] = Distribution1D<NU>(&data[v*nu], nu);
		float marginalFunc[NV];
		for (unsigned int v = 0; v < nv; ++v)
			marginalFunc[v] = pConditionalV[v].funcInt;
		pMarginal = Distribution1D<NV>(&marginalFunc[0], nv);
	}

	CUDA_FUNC_IN void SampleContinuous(float u0, float u1, float uv[2], float *pdf)
	{
		float pdfs[2];
        int v;
        uv[1] = pMarginal.SampleContinuous(u1, &pdfs[1], &v);
        uv[0] = pConditionalV[v].SampleContinuous(u0, &pdfs[0]);
        *pdf = pdfs[0] * pdfs[1];
	}

	CUDA_FUNC_IN float Pdf(float u, float v) const
	{
		int iu = clamp(Float2Int(u * pConditionalV[0].count), 0, pConditionalV[0].count-1);
        int iv = clamp(Float2Int(v * pMarginal.count), 0, pMarginal.count-1);
        if (pConditionalV[iv].funcInt * pMarginal.funcInt == 0.f)
			return 0.f;
        return (pConditionalV[iv].func[iu] * pMarginal.func[iv]) / (pConditionalV[iv].funcInt * pMarginal.funcInt);
	}
private:
	Distribution1D<NU> pConditionalV[NV];
	Distribution1D<NV> pMarginal;
	unsigned int nu, nv;
};

class MonteCarlo
{
public:
	CUDA_FUNC_IN static float3 SphericalDirection(float theta, float phi)
	{
		float sinTheta, cosTheta, sinPhi, cosPhi;

		sincos(theta, &sinTheta, &cosTheta);
		sincos(phi, &sinPhi, &cosPhi);

		return make_float3(
			sinTheta * cosPhi,
			sinTheta * sinPhi,
			cosTheta
		);
	}

	CUDA_FUNC_IN static float2 toSphericalCoordinates(const float3 &v)
	{
		float2 result = make_float2(
			acos(v.z),
			atan2(v.y, v.x)
		);
		if (result.y < 0)
			result.y += 2*PI;
		return result;
	}

	CUDA_FUNC_IN static bool solveLinearSystem2x2(const float a[2][2], const float b[2], float x[2])
	{
		float det = a[0][0] * a[1][1] - a[0][1] * a[1][0];

		if (abs(det) <= RCPOVERFLOW)
			return false;

		float inverse = (float) 1.0f / det;

		x[0] = (a[1][1] * b[0] - a[0][1] * b[1]) * inverse;
		x[1] = (a[0][0] * b[1] - a[1][0] * b[0]) * inverse;

		return true;
	}

	CUDA_FUNC_IN static void stratifiedSample1D(CudaRNG& random, float *dest, int count, bool jitter)
	{
		float invCount = 1.0f / count;

		for (int i=0; i<count; i++) {
			float offset = jitter ? random.randomFloat() : 0.5f;
			*dest++ = (i + offset) * invCount;
		}
	}

	CUDA_FUNC_IN static void stratifiedSample2D(CudaRNG& random, float2 *dest, int countX, int countY, bool jitter)
	{
		float invCountX = 1.0f / countX;
		float invCountY = 1.0f / countY;

		for (int x=0; x<countX; x++) {
			for (int y=0; y<countY; y++) {
				float offsetX = jitter ? random.randomFloat() : 0.5f;
				float offsetY = jitter ? random.randomFloat() : 0.5f;
				*dest++ = make_float2(
					(x + offsetX) * invCountX,
					(y + offsetY) * invCountY
				);
			}
		}
	}

	CUDA_FUNC_IN static void latinHypercube(CudaRNG& random, float *dest, unsigned int nSamples, size_t nDim)
	{
		float delta = 1 / (float) nSamples;
		for (size_t i = 0; i < nSamples; ++i)
			for (size_t j = 0; j < nDim; ++j)
				dest[nDim * i + j] = (i + random.randomFloat()) * delta;
		for (size_t i = 0; i < nDim; ++i) {
			for (size_t j = 0; j < nSamples; ++j) {
				unsigned int other = Floor2Int(float(nSamples) * random.randomFloat());
				swapk(dest + nDim * j + i, dest + nDim * other + i);
			}
		}
	}

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
		float scale = (cosThetaI_ > 0) ? 1.0f/eta : eta,
			  cosThetaTSqr = 1.0f - (1.0f-cosThetaI_*cosThetaI_) * (scale*scale);

		/* Check for total internal reflection */
		if (cosThetaTSqr <= 0.0f) {
			cosThetaT_ = 0.0f;
			return 1.0f;
		}

		/* Find the absolute cosines of the incident/transmitted rays */
		float cosThetaI = abs(cosThetaI_);
		float cosThetaT = sqrtf(cosThetaTSqr);

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
			  sinThetaI2 = 1-cosThetaI2,
			  sinThetaI4 = sinThetaI2*sinThetaI2;

		float temp1 = eta*eta - k*k - sinThetaI2,
			  a2pb2 = sqrtf(temp1*temp1 + 4*k*k*eta*eta),
			  a     = sqrtf(0.5f * (a2pb2 + temp1));

		float term1 = a2pb2 + cosThetaI2,
			  term2 = 2*a*cosThetaI;

		float Rs2 = (term1 - term2) / (term1 + term2);

		float term3 = a2pb2*cosThetaI2 + sinThetaI4,
			  term4 = term2*sinThetaI2;

		float Rp2 = Rs2 * (term3 - term4) / (term3 + term4);

		return 0.5f * (Rp2 + Rs2);
	}

	CUDA_FUNC_IN static Spectrum fresnelConductorExact(float cosThetaI, const Spectrum &eta, const Spectrum &k) {
		/* Modified from "Optics" by K.D. Moeller, University Science Books, 1988 */

		float cosThetaI2 = cosThetaI*cosThetaI,
			  sinThetaI2 = 1-cosThetaI2,
			  sinThetaI4 = sinThetaI2*sinThetaI2;

		Spectrum temp1 = eta*eta - k*k - Spectrum(sinThetaI2),
			a2pb2 = (temp1*temp1 + k*k*eta*eta*4).safe_sqrt(),
			a     = ((a2pb2 + temp1) * 0.5f).safe_sqrt();

		Spectrum term1 = a2pb2 + Spectrum(cosThetaI2),
				 term2 = a*(2*cosThetaI);

		Spectrum Rs2 = (term1 - term2) / (term1 + term2);

		Spectrum term3 = a2pb2*cosThetaI2 + Spectrum(sinThetaI4),
				 term4 = term2*sinThetaI2;

		Spectrum Rp2 = Rs2 * (term3 - term4) / (term3 + term4);

		return 0.5f * (Rp2 + Rs2);
	}

	CUDA_FUNC_IN static float3 reflect(const float3 &wi, const float3 &n) {
		return 2 * dot(wi, n) * (n) - wi;
	}

	CUDA_FUNC_IN static float3 refract(const float3 &wi, const float3 &n, float eta, float cosThetaT) {
		if (cosThetaT < 0)
			eta = 1.0f / eta;

		return n * (dot(wi, n) * eta + cosThetaT) - wi * eta;
	}

	CUDA_FUNC_IN static float3 refract(const float3 &wi, const float3 &n, float eta) {
		if (eta == 1)
			return -1.0f * wi;

		float cosThetaI = dot(wi, n);
		if (cosThetaI > 0)
			eta = 1.0f / eta;

		/* Using Snell's law, calculate the squared sine of the
		   angle between the normal and the transmitted ray */
		float cosThetaTSqr = 1.0f - (1.0f-cosThetaI*cosThetaI) * (eta*eta);

		/* Check for total internal reflection */
		if (cosThetaTSqr <= 0.0f)
			return make_float3(0.0f);

		return n * (cosThetaI * eta - math::signum(cosThetaI) * sqrtf(cosThetaTSqr)) - wi * eta;
	}

	CUDA_FUNC_IN static float3 refract(const float3 &wi, const float3 &n, float eta, float &cosThetaT, float &F) {
		float cosThetaI = dot(wi, n);
		F = fresnelDielectricExt(cosThetaI, cosThetaT, eta);

		if (F == 1.0f) /* Total internal reflection */
			return make_float3(0.0f);

		if (cosThetaT < 0)
			eta = 1 / eta;

		return n * (eta * cosThetaI + cosThetaT) - wi * eta;
	}

	CUDA_FUNC_IN static float fresnelDielectricExt(float cosThetaI, float eta) { float cosThetaT;
	return MonteCarlo::fresnelDielectricExt(cosThetaI, cosThetaT, eta); }
};