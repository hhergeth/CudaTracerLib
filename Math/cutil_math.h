#ifndef CUTIL_MATH_H
#define CUTIL_MATH_H

#include "cuda_runtime.h"
#include "..\Defines.h"

#define ISBLACK(v) (fsumf(v) == 0.0f)
#define BADFLOAT(x) ((*(uint*)&x & 0x7f000000) == 0x7f000000)
#define PI     3.14159265358979f
#define INV_PI (1.0f / PI)
#define INV_TWOPI (1.0f / (2.0f * PI))
#define INV_FOURPI (1.0f / (4.0f * PI))
#define SQRT_TWO      1.41421356237309504880f
#define INV_SQRT_TWO  0.70710678118654752440f
#define ONE_MINUS_EPS 0.999999940395355225f
#define RCPOVERFLOW   2.93873587705571876e-39f
#define DeltaEpsilon 1e-3f

// The maximum possible value for a 32-bit floating point variable
#ifndef  FLT_MAX
#define  FLT_MAX   ((float)3.40282347e+38) 
#endif

#ifndef isnan
#define isnan(x) (x != x)
#endif

// When ray tracing, the epsilon that t > EPSILON in order to avoid self intersections
#define EPSILON       2e-5f

template<typename T> CUDA_FUNC_IN void swapk(T* a, T* b)
{
	T q = *a;
	*a = *b;
	*b = q;
}

template<typename T> CUDA_FUNC_IN void swapk(T& a, T& b)
{
	T q = a;
	a = b;
	b = q;
}

template<typename T> CUDA_FUNC_IN T MIN(T q0, T q1)
{
	return q0 < q1 ? q0 : q1;
}

template<typename T> CUDA_FUNC_IN T MAX(T q0, T q1)
{
	return q0 > q1 ? q0 : q1;
}

template<typename T> CUDA_FUNC_IN T MIN(T q0, T q1, T q2)
{
	return MIN(MIN(q0, q1), q2);
}

template<typename T> CUDA_FUNC_IN T MAX(T q0, T q1, T q2)
{
	return MAX(MAX(q0, q1), q2);
}

template<typename T> CUDA_FUNC_IN T MIN(T q0, T q1, T q2, T q3)
{
	return MIN(MIN(q0, q1, q2), q3);
}

template<typename T> CUDA_FUNC_IN T MAX(T q0, T q1, T q2, T q3)
{
	return MAX(MAX(q0, q1, q2), q3);
}

#define DMAX2(A, B) ((A) > (B) ? (A) : (B))
#define DMAX3(A, B, C) DMAX2(DMAX2(A, B), C)
#define DMAX4(A, B, C, D) DMAX2(DMAX3(A, B, C), D)
#define DMAX5(A, B, C, D, E) DMAX2(DMAX4(A, B, C, D), E)
#define DMAX6(A, B, C, D, E, F) DMAX2(DMAX5(A, B, C, D, E), F)
#define DMAX7(A, B, C, D, E, F, G) DMAX2(DMAX6(A, B, C, D, E, F), G)
#define DMAX8(A, B, C, D, E, F, G, H) DMAX2(DMAX7(A, B, C, D, E, F, G), H)
#define DMAX9(A, B, C, D, E, F, G, H, I) DMAX2(DMAX8(A, B, C, D, E, F, G, H), I)

#define RND_UP(VAL, MOD) (VAL + (((VAL) % (MOD)) != 0 ? ((MOD) - ((VAL) % (MOD))) : (0)))
#define RND_16(VAL) RND_UP(VAL, 16)

#ifndef __popc
 CUDA_FUNC_IN int __popc(unsigned int u)
 {
         unsigned int uCount;

         uCount = u
                  - ((u >> 1) & 033333333333)
                  - ((u >> 2) & 011111111111);
         return
           ((uCount + (uCount >> 3))
            & 030707070707) % 63;
 }
#endif

#define VEC_INDEX(v, i, T) *(((T*)&v) + i)
#define VEC_INDEXI(v, i, T) (((T*)&v) + i)
#define FL_INDEX(v, i) VEC_INDEX(v, i, float)
#define I_INDEX(v, i) VEC_INDEX(v, i, int)

////////////////////////////////////////////////////////////////////////////////
typedef unsigned int uint;
typedef unsigned short ushort;

#ifndef __CUDACC__
#include <math.h>

inline float fminf(float a, float b)
{
  return a < b ? a : b;
}

inline float fmaxf(float a, float b)
{
  return a > b ? a : b;
}

/*
inline float2 fmaxf(float2& a, float2& b)
{
	return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}

inline float3 fmaxf(float3& a, float3& b)
{
	return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

inline float4 fmaxf(float4& a, float4& b)
{
	return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}
*/
#endif

#ifdef __CUDA_ARCH__
__device__ __inline__ int   min_min   (int a, int b, int c) { int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   min_max   (int a, int b, int c) { int v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_min   (int a, int b, int c) { int v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_max   (int a, int b, int c) { int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
#else
CUDA_FUNC_IN float __int_as_float(int i) { return *(float*)&i; }
CUDA_FUNC_IN int __float_as_int(float f) { return *(int*)&f; }
CUDA_FUNC_IN int min_min(int a, int b, int c) { return MIN(a,b,c); }
CUDA_FUNC_IN int min_max(int a, int b, int c) { return MAX(MIN(a,b), c); }
CUDA_FUNC_IN int max_min(int a, int b, int c) { return MIN(MAX(a,b), c); }
CUDA_FUNC_IN int max_max(int a, int b, int c) { return MAX(a,b,c); }
#endif

#ifndef __CUDACC__
CUDA_FUNC_IN float exp2f(float f) { return powf(2.0f, f); }
CUDA_FUNC_IN float copysignf(float a, float b)
{
  return __int_as_float((__float_as_int(b) &  0x80000000) | 
                        (__float_as_int(a) & ~0x80000000));
}
#endif

CUDA_FUNC_IN float fmin_fmin (float a, float b, float c) { return __int_as_float(min_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
CUDA_FUNC_IN float fmin_fmax (float a, float b, float c) { return __int_as_float(min_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
CUDA_FUNC_IN float fmax_fmin (float a, float b, float c) { return __int_as_float(max_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
CUDA_FUNC_IN float fmax_fmax (float a, float b, float c) { return __int_as_float(max_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }

CUDA_FUNC_IN float spanBeginKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d) {	return fmax_fmax( fminf(a0,a1), fminf(b0,b1), fmin_fmax(c0, c1, d)); }
CUDA_FUNC_IN float spanEndKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d) {	return fmin_fmin( fmaxf(a0,a1), fmaxf(b0,b1), fmax_fmin(c0, c1, d)); }


CUDA_FUNC_IN float signf(const float f)
{
	return f > 0 ? 1.0f : (f < 0 ? -1.0f : 0.0f);
}

CUDA_FUNC_IN float2 signf(const float2 f)
{
	return make_float2(signf(f.x), signf(f.y));
}

CUDA_FUNC_IN float3 signf(const float3 f)
{
	return make_float3(signf(f.x), signf(f.y), signf(f.z));
}

CUDA_FUNC_IN float4 signf(const float4 f)
{
	return make_float4(signf(f.x), signf(f.y), signf(f.z), signf(f.w));
}

CUDA_FUNC_IN float fsumf(float2& v)
{
	return v.x + v.y;
}

CUDA_FUNC_IN float fsumf(float3& v)
{
	return v.x + v.y + v.z;
}

CUDA_FUNC_IN float fsumf(float4& v)
{
	return v.x + v.y + v.z + v.w;
}

CUDA_FUNC_IN float fminf(float2& v)
{
	return MIN(v.x, v.y);
}

CUDA_FUNC_IN float fminf(float3& v)
{
	return MIN(v.x, MIN(v.y, v.z));
}

CUDA_FUNC_IN float fminf(float4& v)
{
	return MIN(v.x, MIN(v.y, MIN(v.z, v.w)));
}

CUDA_FUNC_IN float fmaxf(float2& v)
{
	return MAX(v.x, v.y);
}

CUDA_FUNC_IN float fmaxf(float3& v)
{
	return MAX(v.x, MAX(v.y, v.z));
}

CUDA_FUNC_IN float fmaxf(float4& v)
{
	return MAX(v.x, MAX(v.y, MAX(v.z, v.w)));
}

CUDA_FUNC_IN float2 fabsf(float2& v)
{
	return make_float2(fabsf(v.x), fabsf(v.y));
}

CUDA_FUNC_IN float3 fabsf(float3& v)
{
	return make_float3(fabsf(v.x), fabsf(v.y), fabsf(v.z));
}

CUDA_FUNC_IN float4 fabsf(float4& v)
{
	return make_float4(fabsf(v.x), fabsf(v.y), fabsf(v.z), fabsf(v.w));
}

CUDA_FUNC_IN float2 exp(const float2& a)
{
	return make_float2(exp(a.x), exp(a.y));
}

CUDA_FUNC_IN float3 exp(const float3& a)
{
	return make_float3(exp(a.x), exp(a.y), exp(a.z));
}

CUDA_FUNC_IN float4 exp(const float4& a)
{
	return make_float4(exp(a.x), exp(a.y), exp(a.z), exp(a.w));
}

CUDA_FUNC_IN float frac(float f)
{
	return f - floorf(f);
}

#ifndef __CUDACC__
CUDA_FUNC_IN void sincos(float f, float* a, float* b)
{
	*a = sin(f);
	*b = cos(f);
}
#endif

template<typename T> CUDA_FUNC_IN T bilerp(const float2& uv, const T& lt, const T& rt, const T& ld, const T& rd)
{
	T a = lt + (rt - lt) * uv.x, b = ld + (rd - ld) * uv.x;
	return a + (b - a) * uv.y;
}

CUDA_FUNC_IN uchar3 NormalizedFloat3ToUchar3(float3& v)
{
#define CNV(x) x * 127.0f + 127.0f
	return make_uchar3(CNV(v.x), CNV(v.y), CNV(v.z));
#undef CNV
}

CUDA_FUNC_IN float3 Uchar3ToNormalizedFloat3(uchar3 v)
{
#define CNV(x) (float(x) - 127.0f) / 127.0f
	return make_float3(CNV(v.x), CNV(v.y), CNV(v.z));
#undef CNV
}

CUDA_FUNC_IN uchar2 NormalizedFloat3ToUchar2(const float3& v)
{
	float theta = (acos(v.z)*(255.0f/PI));
	float phi = (atan2(v.y,v.x)*(255.0f/(2.0f*PI)));
	phi = phi < 0 ? (phi + 255) : phi;
	return make_uchar2((unsigned char)theta, (unsigned char)phi);
}

CUDA_FUNC_IN float3 Uchar2ToNormalizedFloat3(const uchar2 v)
{
	float theta = float(v.x)*(1.0f/255.0f)*PI;
	float phi = float(v.y)*(1.0f/255.0f)*PI*2.0f;
	float sinphi, cosphi, costheta, sintheta;
	sincos(phi, &sinphi, &cosphi);
	sincos(theta, &sintheta, &costheta);
	return make_float3(sintheta*cosphi, sintheta * sinphi, costheta);
}

CUDA_FUNC_IN float3 Uchar2ToNormalizedFloat3(unsigned int lowBits)
{
	return Uchar2ToNormalizedFloat3(make_uchar2(lowBits & 0xff, (lowBits >> 8) & 255));
}

// float functions
////////////////////////////////////////////////////////////////////////////////

// lerp
inline __device__ __host__ float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}

// clamp
inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}

// int2 functions
////////////////////////////////////////////////////////////////////////////////

// negate
inline __host__ __device__ int2 operator-(int2 &a)
{
    return make_int2(-a.x, -a.y);
}

// addition
inline __host__ __device__ int2 operator+(int2 a, int2 b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(int2 &a, int2 b)
{
    a.x += b.x; a.y += b.y;
}

// subtract
inline __host__ __device__ int2 operator-(int2 a, int2 b)
{
    return make_int2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(int2 &a, int2 b)
{
    a.x -= b.x; a.y -= b.y;
}

// multiply
inline __host__ __device__ int2 operator*(int2 a, int2 b)
{
    return make_int2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ int2 operator/(int2 a, int2 b)
{
    return make_int2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ int2 operator/(int2 a, int b)
{
    return make_int2(a.x / b, a.y / b);
}
inline __host__ __device__ int2 operator*(int2 a, int s)
{
    return make_int2(a.x * s, a.y * s);
}
inline __host__ __device__ int2 operator*(int s, int2 a)
{
    return make_int2(a.x * s, a.y * s);
}
inline __host__ __device__ void operator*=(int2 &a, int s)
{
    a.x *= s; a.y *= s;
}
inline __host__ __device__ bool operator==(int2 &a, int2 &b)
{
    return a.x == b.x && a.y == b.y;
}

// float2 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
inline __host__ __device__ float2 make_float2(float s)
{
    return make_float2(s, s);
}
inline __host__ __device__ float2 make_float2(int2 a)
{
    return make_float2(float(a.x), float(a.y));
}

// negate
inline __host__ __device__ float2 operator-(float2 &a)
{
    return make_float2(-a.x, -a.y);
}

// addition
inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(float2 &a, float2 b)
{
    a.x += b.x; a.y += b.y;
}

// subtract
inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(float2 &a, float2 b)
{
    a.x -= b.x; a.y -= b.y;
}

// multiply
inline __host__ __device__ float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ float2 operator*(float2 a, float s)
{
    return make_float2(a.x * s, a.y * s);
}
inline __host__ __device__ float2 operator*(float s, float2 a)
{
    return make_float2(a.x * s, a.y * s);
}
inline __host__ __device__ void operator*=(float2 &a, float s)
{
    a.x *= s; a.y *= s;
}

// divide
inline __host__ __device__ float2 operator/(float2 a, float2 b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ float2 operator/(float2 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ float2 operator/(float s, float2 a)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ void operator/=(float2 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

// lerp
inline __device__ __host__ float2 lerp(float2 a, float2 b, float t)
{
    return a + t*(b-a);
}

// clamp
inline __device__ __host__ float2 clamp(float2 v, float a, float b)
{
    return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}

inline __device__ __host__ float2 clamp(float2 v, float2 a, float2 b)
{
    return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}

// dot product
inline __host__ __device__ float dot(float2 a, float2 b)
{ 
    return a.x * b.x + a.y * b.y;
}

// length
inline __host__ __device__ float length(float2 v)
{
    return sqrtf(dot(v, v));
}

// normalize
inline __host__ __device__ float2 normalize(float2 v)
{
    float invLen = 1.0f / sqrtf(dot(v, v));
    return v * invLen;
}

// floor
inline __host__ __device__ float2 floor(const float2 v)
{
    return make_float2(floor(v.x), floor(v.y));
}


inline __host__ __device__ bool operator==(float2 &a, float2 &b)
{
    return a.x == b.x && a.y == b.y;
}

// float3 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
inline __host__ __device__ float3 make_float3(float s)
{
    return make_float3(s, s, s);
}
inline __host__ __device__ float3 make_float3(float2 a)
{
    return make_float3(a.x, a.y, 0.0f);
}
inline __host__ __device__ float3 make_float3(float2 a, float s)
{
    return make_float3(a.x, a.y, s);
}
inline __host__ __device__ float3 make_float3(float4 a)
{
    return make_float3(a.x, a.y, a.z);  // discards w
}
inline __host__ __device__ float3 make_float3(int3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}

// negate
inline __host__ __device__ float3 operator-(float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

// min
static __inline__ __host__ __device__ float3 fminf(float3 a, float3 b)
{
	return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}

// max
static __inline__ __host__ __device__ float3 fmaxf(float3 a, float3 b)
{
	return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}

// addition
inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

// subtract
inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ void operator-=(float3 &a, float3 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

// multiply
inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ float3 operator*(float3 a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ float3 operator*(float s, float3 a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ void operator*=(float3 &a, float s)
{
    a.x *= s; a.y *= s; a.z *= s;
}

// divide
inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ float3 operator/(float3 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ float3 operator/(float s, float3 a)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ void operator/=(float3 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

// lerp
inline __device__ __host__ float3 lerp(float3 a, float3 b, float t)
{
    return a + t*(b-a);
}

// clamp
inline __device__ __host__ float3 clamp(float3 v, float a, float b)
{
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline __device__ __host__ float3 clamp(float3 v, float3 a, float3 b)
{
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

// dot product
inline __host__ __device__ float dot(float3 a, float3 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// cross product
inline __host__ __device__ float3 cross(float3 a, float3 b)
{ 
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}

// length
inline __host__ __device__ float length(float3 v)
{
    return sqrtf(dot(v, v));
}

// normalize
inline __host__ __device__ float3 normalize(float3 v)
{
    float invLen = 1.0f / sqrtf(dot(v, v));
    return v * invLen;
}

// floor
inline __host__ __device__ float3 floor(const float3 v)
{
    return make_float3(floor(v.x), floor(v.y), floor(v.z));
}


inline __host__ __device__ bool operator==(float3 &a, float3 &b)
{
	return a.x == b.x && a.y == b.y && a.z == b.z;
}
inline __host__ __device__ float2 operator!(float3 &a)
{
	return make_float2(a.x, a.y);
}
inline __host__ __device__ unsigned int operator~(float3 &a)
{
	return a.x != 0 || a.y != 0 || a.z != 0;
}


// float4 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
inline __host__ __device__ float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}
inline __host__ __device__ float4 make_float4(float3 a)
{
    return make_float4(a.x, a.y, a.z, 0.0f);
}
inline __host__ __device__ float4 make_float4(float3 a, float w)
{
    return make_float4(a.x, a.y, a.z, w);
}
inline __host__ __device__ float4 make_float4(int4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

// negate
inline __host__ __device__ float4 operator-(float4 &a)
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}

// min
static __inline__ __host__ __device__ float4 fminf(float4 a, float4 b)
{
	return make_float4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w));
}

// max
static __inline__ __host__ __device__ float4 fmaxf(float4 a, float4 b)
{
	return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}

// addition
inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

// subtract
inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(float4 &a, float4 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

// multiply
inline __host__ __device__ float4 operator*(float4 a, float s)
{
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
	return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ float4 operator*(float s, float4 a)
{
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline __host__ __device__ void operator*=(float4 &a, float s)
{
    a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}

// divide
inline __host__ __device__ float4 operator/(float4 a, float4 b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline __host__ __device__ float4 operator/(float4 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ float4 operator/(float s, float4 a)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ void operator/=(float4 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

// lerp
inline __device__ __host__ float4 lerp(float4 a, float4 b, float t)
{
    return a + t*(b-a);
}

// clamp
inline __device__ __host__ float4 clamp(float4 v, float a, float b)
{
    return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

inline __device__ __host__ float4 clamp(float4 v, float4 a, float4 b)
{
    return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

// dot product
inline __host__ __device__ float dot(float4 a, float4 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// length
inline __host__ __device__ float length(float4 r)
{
    return sqrtf(dot(r, r));
}

// normalize
inline __host__ __device__ float4 normalize(float4 v)
{
    float invLen = 1.0f / sqrtf(dot(v, v));
    return v * invLen;
}

// floor
inline __host__ __device__ float4 floor(const float4 v)
{
    return make_float4(floor(v.x), floor(v.y), floor(v.z), floor(v.w));
}
inline __host__ __device__ bool operator==(float4 &a, float4 &b)
{
	return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
inline __host__ __device__ float3 operator!(float4 &a)
{
	return make_float3(a.x, a.y, a.z);
}
inline __host__ __device__ unsigned int operator~(float4 &a)
{
	return a.x != 0 || a.y != 0 || a.z != 0 || a.w != 0;
}

// int3 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
inline __host__ __device__ int3 make_int3(int s)
{
    return make_int3(s, s, s);
}
inline __host__ __device__ int3 make_int3(float3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}

// negate
inline __host__ __device__ int3 operator-(int3 &a)
{
    return make_int3(-a.x, -a.y, -a.z);
}

// addition
inline __host__ __device__ int3 operator+(int3 a, int3 b)
{
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(int3 &a, int3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

// subtract
inline __host__ __device__ int3 operator-(int3 a, int3 b)
{
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ void operator-=(int3 &a, int3 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

// multiply
inline __host__ __device__ int3 operator*(int3 a, int3 b)
{
    return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ int3 operator*(int3 a, int s)
{
    return make_int3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ int3 operator*(int s, int3 a)
{
    return make_int3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ void operator*=(int3 &a, int s)
{
    a.x *= s; a.y *= s; a.z *= s;
}

// divide
inline __host__ __device__ int3 operator/(int3 a, int3 b)
{
    return make_int3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ int3 operator/(int3 a, int s)
{
    return make_int3(a.x / s, a.y / s, a.z / s);
}
inline __host__ __device__ int3 operator/(int s, int3 a)
{
    return make_int3(a.x / s, a.y / s, a.z / s);
}
inline __host__ __device__ void operator/=(int3 &a, int s)
{
    a.x /= s; a.y /= s; a.z /= s;
}

// clamp
inline __device__ __host__ int clamp(int f, int a, int b)
{
    return MAX(a, MIN(f, b));
}

inline __device__ __host__ int3 clamp(int3 v, int a, int b)
{
    return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline __device__ __host__ int3 clamp(int3 v, int3 a, int3 b)
{
    return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __host__ __device__ bool operator==(int3 &a, int3 &b)
{
	return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline __device__ __host__ int2 clamp(int2 v, int a, int b)
{
    return make_int2(clamp(v.x, a, b), clamp(v.y, a, b));
}

inline __device__ __host__ int2 clamp(int2 v, int2 a, int2 b)
{
    return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}


// uint3 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
inline __host__ __device__ uint3 make_uint3(uint s)
{
    return make_uint3(s, s, s);
}
inline __host__ __device__ uint3 make_uint3(float3 a)
{
    return make_uint3(uint(a.x), uint(a.y), uint(a.z));
}
inline __host__ __device__ uint3 make_uint3(dim3 s)
{
    return make_uint3(s.x, s.y, s.z);
}

// addition
inline __host__ __device__ uint3 operator+(uint3 a, uint3 b)
{
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(uint3 &a, uint3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

// subtract
inline __host__ __device__ uint3 operator-(uint3 a, uint3 b)
{
    return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ void operator-=(uint3 &a, uint3 b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

// multiply
inline __host__ __device__ uint3 operator*(uint3 a, uint3 b)
{
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ uint3 operator*(uint3 a, uint s)
{
    return make_uint3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ uint3 operator*(uint s, uint3 a)
{
    return make_uint3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ void operator*=(uint3 &a, uint s)
{
    a.x *= s; a.y *= s; a.z *= s;
}

// divide
inline __host__ __device__ uint3 operator/(uint3 a, uint3 b)
{
    return make_uint3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ uint3 operator/(uint3 a, uint s)
{
    return make_uint3(a.x / s, a.y / s, a.z / s);
}
inline __host__ __device__ uint3 operator/(uint s, uint3 a)
{
    return make_uint3(a.x / s, a.y / s, a.z / s);
}
inline __host__ __device__ void operator/=(uint3 &a, uint s)
{
    a.x /= s; a.y /= s; a.z /= s;
}

// clamp
inline __device__ __host__ uint clamp(uint f, uint a, uint b)
{
	return MAX(a, MIN(f, b));
}

inline __device__ __host__ uint3 clamp(uint3 v, uint a, uint b)
{
    return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline __device__ __host__ uint3 clamp(uint3 v, uint3 a, uint3 b)
{
    return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __host__ __device__ bool operator==(uint3 &a, uint3 &b)
{
	return a.x == b.x && a.y == b.y && a.z == b.z;
}

CUDA_FUNC_IN float2 saturate(const float2& v)
{
	return clamp(v, make_float2(0), make_float2(1));
}

CUDA_FUNC_IN float3 saturate(const float3& v)
{
	return clamp(v, make_float3(0), make_float3(1));
}

CUDA_FUNC_IN float4 saturate(const float4& v)
{
	return clamp(v, make_float4(0), make_float4(1));
}

CUDA_FUNC_IN float2 fsqrtf(const float2& v)
{
	return make_float2(sqrtf(v.x), sqrtf(v.y));
}

CUDA_FUNC_IN float3 fsqrtf(const float3& v)
{
	return make_float3(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z));
}

CUDA_FUNC_IN float4 fsqrtf(const float4& v)
{
	return make_float4(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z), sqrtf(v.w));
}

CUDA_FUNC_IN float DistanceSquared(const float3& a, const float3& b)
{
	float3 q = a - b;
	return dot(q, q);
}

CUDA_FUNC_IN float Distance(const float3& a, const float3& b)
{
	return sqrtf(DistanceSquared(a, b));
}

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

CUDA_FUNC_IN float clamp01(float a)
{
	return clamp(a,0.0f,1.0f);
}

CUDA_FUNC_IN float2 clamp01(const float2& a)
{
	return make_float2(clamp(a.x, 0.0f, 1.0f), clamp(a.y, 0.0f, 1.0f));
}

CUDA_FUNC_IN float3 clamp01(const float3& a)
{
	return make_float3(clamp(a.x, 0.0f, 1.0f), clamp(a.y, 0.0f, 1.0f), clamp(a.z, 0.0f, 1.0f));
}

CUDA_FUNC_IN float4 clamp01(const float4& a)
{
	return make_float4(clamp(a.x, 0.0f, 1.0f), clamp(a.y, 0.0f, 1.0f), clamp(a.z, 0.0f, 1.0f), clamp(a.w, 0.0f, 1.0f));
}

CUDA_FUNC_IN float2 sqrtf(const float2& v)
{
	return make_float2(sqrtf(v.x), sqrtf(v.y));
}

CUDA_FUNC_IN float3 sqrtf(const float3& v)
{
	return make_float3(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z));
}

CUDA_FUNC_IN float4 sqrtf(const float4& v)
{
	return make_float4(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z), sqrtf(v.w));
}

class math
{
public:
	/// Arcsine variant that gracefully handles arguments > 1 that are due to roundoff errors
	CUDA_FUNC_IN static float safe_asin(float value) {
		return asinf(MIN(1.0f, MAX(-1.0f, value)));
	}

	/// Arccosine variant that gracefully handles arguments > 1 that are due to roundoff errors
	CUDA_FUNC_IN static float safe_acos(float value) {
		return acosf(MIN(1.0f, MAX(-1.0f, value)));
	}

	/// Square root variant that gracefully handles arguments < 0 that are due to roundoff errors
	CUDA_FUNC_IN static float safe_sqrt(float value) {
		return sqrtf(MAX(0.0f, value));
	}

	CUDA_FUNC_IN static float signum(float value)
	{
#ifdef ISCUDA
		return copysign(1.0f, value);
#else
		return (float)_copysign(1.0f, value);
#endif
	}
};

class VectorMath
{
public:
	static inline __host__ __device__ float3 refract(const float3& i, const float3& n, const float refrIndex)
	{
			float cosI = -dot( i, n );
			float cosT2 = 1.0f - refrIndex * refrIndex * (1.0f - cosI * cosI);
			float3 t = (refrIndex * i) + (refrIndex * cosI - sqrt( abs(cosT2) )) * n;
			return t * cosT2;
	}
	// reflect
	static inline __host__ __device__ float2 reflect(float2 i, float2 n)
	{
		return i - 2.0f * n * dot(n,i);
	}
	// reflect
	static inline __host__ __device__ float3 reflect(float3 i, float3 n)
	{
		return i - 2.0f * n * dot(n,i);
	}
};

#endif