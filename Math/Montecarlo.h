#pragma once

#include "..\Base\CudaRandom.h"

CUDA_FUNC_IN float AbsDot(const float3& a, const float3& b)
{
	return abs(dot(a, b));
}

CUDA_FUNC_IN int Mod(int a, int b) {
    int n = int(a/b);
    a -= n*b;
    if (a < 0) a += b;
    return a;
}


CUDA_FUNC_IN float Radians(float deg) {
    return ((float)PI/180.f) * deg;
}


CUDA_FUNC_IN float Degrees(float rad) {
    return (180.f/(float)PI) * rad;
}


CUDA_FUNC_IN float Log2(float x) {
    float invLog2 = 1.f / logf(2.f);
    return logf(x) * invLog2;
}

CUDA_FUNC_IN int Floor2Int(float val) {
    return (int)floorf(val);
}

CUDA_FUNC_IN int Log2Int(float v)
{
    return Floor2Int(Log2(v));
}

CUDA_FUNC_IN bool IsPowerOf2(int v) {
    return (v & (v - 1)) == 0;
}

CUDA_FUNC_IN unsigned int RoundUpPow2(unsigned int v) {
    v--;
    v |= v >> 1;    v |= v >> 2;
    v |= v >> 4;    v |= v >> 8;
    v |= v >> 16;
    return v+1;
}


CUDA_FUNC_IN int Round2Int(float val) {
    return Floor2Int(val + 0.5f);
}


CUDA_FUNC_IN int Float2Int(float val) {
    return (int)val;
}


CUDA_FUNC_IN int Ceil2Int(float val) {
    return (int)ceilf(val);
}


#ifdef NDEBUG
#define Assert(expr) ((void)0)
#else
#define Assert(expr) ((void)0)
/*
#define Assert(expr) \
    ((expr) ? (void)0 : \
        Severe("Assertion \"%s\" failed in %s, line %d", \
               #expr, __FILE__, __LINE__))
*/
#endif // NDEBUG

template<typename T> CUDA_FUNC_IN void swap(T* a, T* b)
{
	T q = *a;
	*a = *b;
	*b = q;
}

CUDA_FUNC_IN bool Quadratic(float A, float B, float C, float *t0, float *t1)
{
    // Find quadratic discriminant
    float discrim = B * B - 4.f * A * C;
    if (discrim <= 0.) return false;
    float rootDiscrim = sqrtf(discrim);

    // Compute quadratic _t_ values
    float q;
    if (B < 0) q = -.5f * (B - rootDiscrim);
    else       q = -.5f * (B + rootDiscrim);
    *t0 = q / A;
    *t1 = C / q;
    if (*t0 > *t1)
		swap(t0, t1);
    return true;
}

#define OneMinusEpsilon 0.9999999403953552f


CUDA_ONLY_FUNC void RejectionSampleDisk(float *x, float *y, CudaRNG &rng)
{
	float sx, sy;
    do {
        sx = 1.f - 2.f * rng.randomFloat();
        sy = 1.f - 2.f * rng.randomFloat();
    } while (sx*sx + sy*sy > 1.f);
    *x = sx;
    *y = sy;
}
CUDA_FUNC_IN float3 UniformSampleHemisphere(float u1, float u2)
{
    float z = u1;
    float r = sqrtf(MAX(0.f, 1.f - z*z));
    float phi = 2 * PI * u2;
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    return make_float3(x, y, z);
}
CUDA_FUNC_IN float  UniformHemispherePdf()
{
	return 1.0f / (2.0f * PI);
}
CUDA_FUNC_IN float3 UniformSampleSphere(float u1, float u2)
{
	float z = 1.f - 2.f * u1;
    float r = sqrtf(MAX(0.f, 1.f - z*z));
    float phi = 2.f * PI * u2;
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    return make_float3(x, y, z);
}
CUDA_FUNC_IN float  UniformSpherePdf()
{
	return 1.f / (4.f * PI);
}
CUDA_FUNC_IN float3 UniformSampleCone(float u1, float u2, float costhetamax)
{
	float costheta = (1.f - u1) + u1 * costhetamax;
    float sintheta = sqrtf(1.f - costheta*costheta);
    float phi = u2 * 2.f * PI;
    return make_float3(cosf(phi) * sintheta, sinf(phi) * sintheta, costheta);
}
CUDA_FUNC_IN float3 UniformSampleCone(float u1, float u2, float costhetamax, const float3 &x, const float3 &y, const float3 &z)
{
    float costheta = lerp(costhetamax, 1.f, u1);
    float sintheta = sqrtf(1.f - costheta*costheta);
    float phi = u2 * 2.f * PI;
    return cosf(phi) * sintheta * x + sinf(phi) * sintheta * y + costheta * z;
}
CUDA_FUNC_IN float  UniformConePdf(float cosThetaMax)
{
	return 1.f / (2.f * PI * (1.f - cosThetaMax));
}
CUDA_FUNC_IN void UniformSampleDisk(float u1, float u2, float *x, float *y)
{
	float r = sqrtf(u1);
    float theta = 2.0f * PI * u2;
    *x = r * cosf(theta);
    *y = r * sinf(theta);
}
CUDA_FUNC_IN void ConcentricSampleDisk(float u1, float u2, float *dx, float *dy)
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
            if (sy > 0.0) theta = sy/r;
            else          theta = 8.0f + sy/r;
        }
        else {
            // Handle second region of disk
            r = sy;
            theta = 2.0f - sx/r;
        }
    }
    else {
        if (sx <= sy) {
            // Handle third region of disk
            r = -sx;
            theta = 4.0f - sy/r;
        }
        else {
            // Handle fourth region of disk
            r = -sy;
            theta = 6.0f + sx/r;
        }
    }
    theta *= PI / 4.f;
    *dx = r * cosf(theta);
    *dy = r * sinf(theta);
}
CUDA_FUNC_IN float3 CosineSampleHemisphere(float u1, float u2) {
    float3 ret;
    ConcentricSampleDisk(u1, u2, &ret.x, &ret.y);
    ret.z = sqrtf(MAX(0.f, 1.f - ret.x*ret.x - ret.y*ret.y));
    return ret;
}
CUDA_FUNC_IN float CosineHemispherePdf(float costheta, float phi)
{
    return costheta / PI;
}
CUDA_FUNC_IN void StratifiedSample1D(float *samples, int nSamples, CudaRNG &rng, bool jitter = true)
{
    float invTot = 1.f / nSamples;
    for (int i = 0;  i < nSamples; ++i)
	{
        float delta = jitter ? rng.randomFloat() : 0.5f;
        *samples++ = MIN((i + delta) * invTot, OneMinusEpsilon);
    }
}
CUDA_FUNC_IN void StratifiedSample2D(float *samples, int nx, int ny, CudaRNG &rng, bool jitter = true)
{
    float dx = 1.f / nx, dy = 1.f / ny;
    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x)
		{
            float jx = jitter ? rng.randomFloat() : 0.5f;
            float jy = jitter ? rng.randomFloat() : 0.5f;
			*samples++ = MIN((x + jx) * dx, OneMinusEpsilon);
			*samples++ = MIN((y + jy) * dy, OneMinusEpsilon);
        }
}
template <typename T> CUDA_ONLY_FUNC void Shuffle(T *samp, unsigned int count, unsigned int dims, CudaRNG &rng)
{
    for (unsigned int i = 0; i < count; ++i)
	{
        unsigned int other = i + (rng.randomUint() % (count - i));
        for (unsigned int j = 0; j < dims; ++j)
            swap(samp[dims*i + j], samp[dims*other + j]);
    }
}

CUDA_FUNC_IN float3 FrDiel(float cosi, float cost, const float3 &etai, const float3 &etat)
{
    float3 Rparl = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
    float3 Rperp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
    return (Rparl*Rparl + Rperp*Rperp) / 2.f;
}

CUDA_FUNC_IN float3 FrCond(float cosi, const float3 &eta, const float3 &k)
{
    float3 tmp = (eta*eta + k*k) * cosi*cosi;
    float3 Rparl2 = (tmp - (2.f * eta * cosi) + 1) / (tmp + (2.f * eta * cosi) + 1);
    float3 tmp_f = eta*eta + k*k;
    float3 Rperp2 = (tmp_f - (2.f * eta * cosi) + cosi*cosi) / (tmp_f + (2.f * eta * cosi) + cosi*cosi);
    return (Rparl2 + Rperp2) / 2.f;
}

CUDA_FUNC_IN float Fdr(float eta)
{
    if (eta >= 1)
        return -1.4399f / (eta*eta) + 0.7099f / eta + 0.6681f + 0.0636f * eta;
    else return -0.4399f + .7099f / eta - .3319f / (eta * eta) + .0636f / (eta*eta*eta);
}

CUDA_FUNC_IN float CosTheta(const float3 &w)
{
	return w.z;
}

CUDA_FUNC_IN float AbsCosTheta(const float3 &w)
{
	return fabsf(w.z);
}

CUDA_FUNC_IN float SinTheta2(const float3 &w) 
{
    return MAX(0.f, 1.f - CosTheta(w)*CosTheta(w));
}

CUDA_FUNC_IN float SinTheta(const float3 &w)
{
    return sqrtf(SinTheta2(w));
}

CUDA_FUNC_IN float CosPhi(const float3 &w)
{
    float sintheta = SinTheta(w);
    if (sintheta == 0.f)
		return 1.f;
    return clamp(w.x / sintheta, -1.f, 1.f);
}

CUDA_FUNC_IN float SinPhi(const float3 &w)
{
    float sintheta = SinTheta(w);
    if (sintheta == 0.f) return 0.f;
    return clamp(w.y / sintheta, -1.f, 1.f);
}

CUDA_FUNC_IN bool SameHemisphere(const float3 &w, const float3 &wp)
{
    return w.z * wp.z > 0.0f;
}

CUDA_FUNC_IN float3 SphericalDirection(float sintheta, float costheta, float phi)
{
    return make_float3(sintheta * cosf(phi),
                  sintheta * sinf(phi),
                  costheta);
}

CUDA_FUNC_IN float3 SphericalDirection(float sintheta, float costheta, float phi, const float3 &x, const float3 &y, const float3 &z)
{
    return sintheta * cosf(phi) * x +
           sintheta * sinf(phi) * y + costheta * z;
}

CUDA_FUNC_IN float SphericalTheta(const float3 &v)
{
    return acosf(clamp(-1.f, 1.f, v.z));
}

CUDA_FUNC_IN float SphericalPhi(const float3 &v)
{
    float p = atan2f(v.y, v.x);
    return (p < 0.f) ? p + 2.f * PI : p;
}