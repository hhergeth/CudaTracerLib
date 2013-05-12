#pragma once

#include "..\Base\CudaRandom.h"

struct LightSample
{
	CUDA_FUNC_IN LightSample() { }
	CUDA_FUNC_IN LightSample(CudaRNG &rng)
	{
		uPos[0] = rng.randomFloat();
		uPos[1] = rng.randomFloat();
		uComponent = rng.randomFloat();
	}
	CUDA_FUNC_IN LightSample(float up0, float up1, float ucomp)
	{
		uPos[0] = up0; uPos[1] = up1;
		uComponent = ucomp;
	}
	float uPos[2], uComponent;
};

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

CUDA_FUNC_IN void UniformSampleTriangle(float u1, float u2, float *u, float *v)
{
    float su1 = sqrtf(u1);
    *u = 1.f - su1;
    *v = u2 * su1;
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

CUDA_FUNC_IN float BalanceHeuristic(int nf, float fPdf, int ng, float gPdf)
{
    return (nf * fPdf) / (nf * fPdf + ng * gPdf);
}


CUDA_FUNC_IN float PowerHeuristic(int nf, float fPdf, int ng, float gPdf)
{
    float f = nf * fPdf, g = ng * gPdf;
    return (f*f) / (f*f + g*g);
}

template<typename T> CUDA_FUNC_IN void STL_Sort(T* a_Array, unsigned int a_Length, int (*cmp)(T*, T*))
{
	for (int i = 0; i < a_Length -1; ++i)
		for (int j = 0; j < a_Length - i - 1; ++j)
			if (cmp(a_Array[j], a_Array[j + 1]) > 0) 
				swap(a_Array[j], a_Array[j + 1]);
}



template<typename T> CUDA_FUNC_IN void STL_Sort(T* a_Array, unsigned int a_Length)
{
	for (int i = 0; i < a_Length -1; ++i)
		for (int j = 0; j < a_Length - i - 1; ++j)
			if (a_Array[j] > a_Array[j + 1]) 
				swap(a_Array[j], a_Array[j + 1]);
}

template<typename T> CUDA_FUNC_IN const T* STL_upper_bound(const T* first, const T* last, const T& value)
{
	const T* f = first;
	do
	{
		if(*f > value)
			return f;
	}
	while(f++ <= last);
	return first;
}

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
        float *ptr = STL_upper_bound(cdf, cdf+count+1, u);
        int offset = MAX(0, int(ptr-cdf-1));
        if (off)
			*off = offset;
        Assert(offset < count);
        Assert(u >= cdf[offset] && u < cdf[offset+1]);

        // Compute offset along CDF segment
        float du = (u - cdf[offset]) / (cdf[offset+1] - cdf[offset]);
        Assert(!isnan(du));

        // Compute PDF for sampled offset
        if (pdf) *pdf = func[offset] / funcInt;

        // Return $x\in{}[0,1)$ corresponding to sample
        return (offset + du) / count;
    }
    CUDA_FUNC_IN int SampleDiscrete(float u, float *pdf) const
	{
        const float *ptr = STL_upper_bound(cdf, cdf+count+1, u);
		int offset = MAX(0, int(ptr-cdf-1));
        Assert(offset < count);
        Assert(u >= cdf[offset] && u < cdf[offset+1]);
        if (pdf) *pdf = func[offset] / (funcInt * count);
        return offset;
    }
private:
    float func[N];
	float cdf[N];
    float funcInt;
    int count;
};

CUDA_FUNC_IN float RdIntegral(float alphap, float A)
{
    float sqrtTerm = sqrtf(3.f * (1.f - alphap));
    return alphap / 2.f * (1.f + expf(-4.f/3.f * A * sqrtTerm)) * expf(-sqrtTerm);
}

CUDA_FUNC_IN float RdToAlphap(float reflectance, float A)
{
    float alphaLow = 0., alphaHigh = 1.f;
    float kd0 = RdIntegral(alphaLow, A);
    float kd1 = RdIntegral(alphaHigh, A);
    for (int i = 0; i < 16; ++i) {
        Assert(kd0 <= reflectance && kd1 >= reflectance);
        float alphaMid = (alphaLow + alphaHigh) * 0.5f;
        float kd = RdIntegral(alphaMid, A);
        if (kd < reflectance) { alphaLow = alphaMid;  kd0 = kd; }
        else                  { alphaHigh = alphaMid; kd1 = kd; }
    }
    return (alphaLow + alphaHigh) * 0.5f;
}

CUDA_FUNC_IN void SubsurfaceFromDiffuse(const float3 &Kd, float meanPathLength, float eta, float3 *sigma_a, float3 *sigma_prime_s)
{
    float A = (1.f + Fdr(eta)) / (1.f - Fdr(eta));
    float rgb[3];
    rgb[0] = Kd.x; rgb[1] = Kd.y; rgb[2] = Kd.z;
    float sigma_prime_s_rgb[3], sigma_a_rgb[3];
    for (int i = 0; i < 3; ++i)
	{
       // Compute $\alpha'$ for RGB component, compute scattering properties
       float alphap = RdToAlphap(rgb[i], A);
       float sigma_tr = 1.f / meanPathLength;
       float sigma_prime_t = sigma_tr / sqrtf(3.f * 1.f - alphap);
       sigma_prime_s_rgb[i] = alphap * sigma_prime_t;
       sigma_a_rgb[i] = sigma_prime_t - sigma_prime_s_rgb[i];
    }
    *sigma_a = make_float3(sigma_a_rgb[0], sigma_a_rgb[1], sigma_a_rgb[2]);
    *sigma_prime_s = make_float3(sigma_prime_s_rgb[0], sigma_prime_s_rgb[1], sigma_prime_s_rgb[2]);
}