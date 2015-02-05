#pragma once

#include "../Defines.h"
#include <math.h>
#include <float.h>

#define PI     3.14159265358979f
#define INV_PI (1.0f / PI)
#define INV_TWOPI (1.0f / (2.0f * PI))
#define INV_FOURPI (1.0f / (4.0f * PI))
#define SQRT_TWO      1.41421356237309504880f
#define INV_SQRT_TWO  0.70710678118654752440f
#define ONE_minUS_EPS 0.999999940395355225f
#define EPSILON 0.000001f
#define RCPOVERFLOW   2.93873587705571876e-39f
#define DeltaEpsilon 1e-3f

#ifndef isnan
#define isnan(x) (x != x)
#endif

CUDA_FUNC_IN float int_as_float_(int i)
{
#ifdef ISCUDA

	return __int_as_float(i);
#else
	return *(float*)&i;
#endif
}

CUDA_FUNC_IN int float_as_int_(float f)
{
#ifdef ISCUDA
	return __float_as_int(f);
#else
	return *(int*)&f;;
#endif
}

CUDA_FUNC_IN int popc(unsigned int u)
{
#ifdef ISCUDA
	return __popc(u);
#else
	unsigned int uCount;

	uCount = u
		- ((u >> 1) & 033333333333)
		- ((u >> 2) & 011111111111);
	return
		((uCount + (uCount >> 3))
		& 030707070707) % 63;
#endif
}

#ifndef __CUDACC__
CUDA_FUNC_IN float copysignf(float a, float b)
{
	return int_as_float_((float_as_int_(b) & 0x80000000) | (float_as_int_(a) & ~0x80000000));
}
CUDA_FUNC_IN void sincos(float f, float* a, float* b)
{
	*a = sin(f);
	*b = cos(f);
}
#endif

#define FW_SPECIALIZE_MINMAX(TEMPLATE, T, K, MIN, MAX) \
    TEMPLATE CUDA_FUNC_IN T min(K T& a, K T& b) { return MIN; } \
    TEMPLATE CUDA_FUNC_IN T max(K T& a, K T& b) { return MAX; } \
    TEMPLATE CUDA_FUNC_IN T min(K T& a, K T& b, K T& c) { return min(min(a, b), c); } \
    TEMPLATE CUDA_FUNC_IN T max(K T& a, K T& b, K T& c) { return max(max(a, b), c); } \
    TEMPLATE CUDA_FUNC_IN T min(K T& a, K T& b, K T& c, K T& d) { return min(min(min(a, b), c), d); } \
    TEMPLATE CUDA_FUNC_IN T max(K T& a, K T& b, K T& c, K T& d) { return max(max(max(a, b), c), d); } \
    TEMPLATE CUDA_FUNC_IN T min(K T& a, K T& b, K T& c, K T& d, K T& e) { return min(min(min(min(a, b), c), d), e); } \
    TEMPLATE CUDA_FUNC_IN T max(K T& a, K T& b, K T& c, K T& d, K T& e) { return max(max(max(max(a, b), c), d), e); } \
    TEMPLATE CUDA_FUNC_IN T min(K T& a, K T& b, K T& c, K T& d, K T& e, K T& f) { return min(min(min(min(min(a, b), c), d), e), f); } \
    TEMPLATE CUDA_FUNC_IN T max(K T& a, K T& b, K T& c, K T& d, K T& e, K T& f) { return max(max(max(max(max(a, b), c), d), e), f); } \
    TEMPLATE CUDA_FUNC_IN T min(K T& a, K T& b, K T& c, K T& d, K T& e, K T& f, K T& g) { return min(min(min(min(min(min(a, b), c), d), e), f), g); } \
    TEMPLATE CUDA_FUNC_IN T max(K T& a, K T& b, K T& c, K T& d, K T& e, K T& f, K T& g) { return max(max(max(max(max(max(a, b), c), d), e), f), g); } \
    TEMPLATE CUDA_FUNC_IN T min(K T& a, K T& b, K T& c, K T& d, K T& e, K T& f, K T& g, K T& h) { return min(min(min(min(min(min(min(a, b), c), d), e), f), g), h); } \
    TEMPLATE CUDA_FUNC_IN T max(K T& a, K T& b, K T& c, K T& d, K T& e, K T& f, K T& g, K T& h) { return max(max(max(max(max(max(max(a, b), c), d), e), f), g), h); } \

FW_SPECIALIZE_MINMAX(template <class T>, T, , (a < b) ? a : b, (a > b) ? a : b)
FW_SPECIALIZE_MINMAX(template <class T>, T, const, (a < b) ? a : b, (a > b) ? a : b)

class math
{
public:
	/// Arcsine variant that gracefully handles arguments > 1 that are due to roundoff errors
	CUDA_FUNC_IN static float safe_asin(float value) {
		return asinf(min(1.0f, max(-1.0f, value)));
	}

	/// Arccosine variant that gracefully handles arguments > 1 that are due to roundoff errors
	CUDA_FUNC_IN static float safe_acos(float value) {
		return acosf(min(1.0f, max(-1.0f, value)));
	}

	/// Square root variant that gracefully handles arguments < 0 that are due to roundoff errors
	CUDA_FUNC_IN static float safe_sqrt(float value) {
		return sqrt(max(0.0f, value));
	}

	CUDA_FUNC_IN static float signum(float value)
	{
		return copysign(1.0f, value);
	}

	/// Always-positive modulo function (assumes b > 0)
	CUDA_FUNC_IN static int modulo(int a, int b)
	{
		int r = a % b;
		return (r < 0) ? r + b : r;
	}

	/// Always-positive modulo function, float version (assumes b > 0)
	CUDA_FUNC_IN static float modulo(float a, float b)
	{
		float r = fmod(a, b);
		return (r < 0) ? r + b : r;
	}

	CUDA_FUNC_IN static int min_min(int a, int b, int c)
	{
#ifdef ISCUDA
		int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v;
#else
		return min(a, b, c);
#endif
	}
	CUDA_FUNC_IN static int min_max(int a, int b, int c)
	{
#ifdef ISCUDA
		int v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v;
#else
		return max(min(a, b), c);
#endif
	}
	CUDA_FUNC_IN static int max_min(int a, int b, int c)
	{
#ifdef ISCUDA
		int v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v;
#else
		return min(max(a, b), c);
#endif
	}
	CUDA_FUNC_IN static int max_max(int a, int b, int c)
	{
#ifdef ISCUDA
		int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v;
#else
		return max(a, b, c);
#endif
	}

	CUDA_FUNC_IN static float fmin_fmin(float a, float b, float c) { return int_as_float_(min_min(float_as_int_(a), float_as_int_(b), float_as_int_(c))); }
	CUDA_FUNC_IN static float fmin_fmax(float a, float b, float c) { return int_as_float_(min_max(float_as_int_(a), float_as_int_(b), float_as_int_(c))); }
	CUDA_FUNC_IN static float fmax_fmin(float a, float b, float c) { return int_as_float_(max_min(float_as_int_(a), float_as_int_(b), float_as_int_(c))); }
	CUDA_FUNC_IN static float fmax_fmax(float a, float b, float c) { return int_as_float_(max_max(float_as_int_(a), float_as_int_(b), float_as_int_(c))); }

	CUDA_FUNC_IN static float spanBeginKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d) { return fmax_fmax(min(a0, a1), min(b0, b1), fmin_fmax(c0, c1, d)); }
	CUDA_FUNC_IN static float spanEndKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d) { return fmin_fmin(max(a0, a1), max(b0, b1), fmax_fmin(c0, c1, d)); }

	template<typename T> CUDA_FUNC_IN static T sign(T f)
	{
		return f > T(0) ? T(1) : (f < T(0) ? T(-1) : T(0));
	}

	CUDA_FUNC_IN static float frac(float f)
	{
		return f - floorf(f);
	}

	template<typename T> CUDA_FUNC_IN static T bilerp(const float2& uv, const T& lt, const T& rt, const T& ld, const T& rd)
	{
		T a = lt + (rt - lt) * uv.x, b = ld + (rd - ld) * uv.x;
		return a + (b - a) * uv.y;
	}

	CUDA_FUNC_IN static int Floor2Int(float val) {
		return (int)floorf(val);
	}
	CUDA_FUNC_IN static int Round2Int(float val) {
		return Floor2Int(val + 0.5f);
	}
	CUDA_FUNC_IN static int Float2Int(float val) {
		return (int)val;
	}
	CUDA_FUNC_IN static int Ceil2Int(float val) {
		return (int)ceilf(val);
	}

	CUDA_FUNC_IN static float lerp(float a, float b, float t)
	{
		return a + t*(b - a);
	}

	template <class T> CUDA_FUNC_IN static T clamp(T& v, T& lo, T& hi) { return min(max(v, lo), hi); }
	template <class T> CUDA_FUNC_IN static T clamp(const T& v, const T& lo, const T& hi) { return min(max(v, lo), hi); }
	template <class T> CUDA_FUNC_IN static T clamp01(T& a) { return clamp(a, T(0.0f), T(1.0f)); }
	template <class T> CUDA_FUNC_IN static T clamp01(const T& a) { return clamp(a, T(0.0f), T(1.0f)); }

	CUDA_FUNC_IN static int Mod(int a, int b) {
		int n = int(a / b);
		a -= n*b;
		if (a < 0) a += b;
		return a;
	}

	CUDA_FUNC_IN static float Radians(float deg) {
		return ((float)PI / 180.f) * deg;
	}

	CUDA_FUNC_IN static float Degrees(float rad) {
		return (180.f / (float)PI) * rad;
	}

	CUDA_FUNC_IN static float Log2(float x) {
		float invLog2 = 1.f / logf(2.f);
		return logf(x) * invLog2;
	}

	CUDA_FUNC_IN static int Log2Int(float v)
	{
		return Floor2Int(Log2(v));
	}

	CUDA_FUNC_IN static bool IsPowerOf2(int v) {
		return (v & (v - 1)) == 0;
	}

	CUDA_FUNC_IN static unsigned int RoundUpPow2(unsigned int v) {
		v--;
		v |= v >> 1;    v |= v >> 2;
		v |= v >> 4;    v |= v >> 8;
		v |= v >> 16;
		return v + 1;
	}

	CUDA_FUNC_IN static float variance(float x, float x2, float n)
	{
		return x2 / n - x / n * x / n;
	}

	CUDA_FUNC_IN static float    sqrt(float a)         { return ::sqrt(a); }
	CUDA_FUNC_IN static int    abs(int a)         { return (a >= 0) ? a : -a; }
	CUDA_FUNC_IN static float    abs(float a)         { return ::fabsf(a); }
	CUDA_FUNC_IN static float    asin(float a)         { return ::asinf(a); }
	CUDA_FUNC_IN static float    acos(float a)         { return ::acosf(a); }
	CUDA_FUNC_IN static float    atan(float a)         { return ::atanf(a); }
	CUDA_FUNC_IN static float    atan2(float y, float x)  { return ::atan2f(y, x); }
	CUDA_FUNC_IN static float    floor(float a)         { return ::floorf(a); }
	CUDA_FUNC_IN static float    ceil(float a)         { return ::ceilf(a); }

	CUDA_FUNC_IN static float pow(float a, float b)
	{
#ifdef ISCUDA
		return ::__powf(a, b);
#else
		return ::powf(a, b);
#endif
	}

	CUDA_FUNC_IN static float exp(float a)
	{
#ifdef ISCUDA
		return ::__expf(a);
#else
		return ::expf(a);
#endif
	}

	CUDA_FUNC_IN static float exp2(float a)
	{
#ifdef ISCUDA
		return ::exp2f(a);
#else
		return ::powf(2.0f, a);
#endif
	}

	CUDA_FUNC_IN static float log(float a)
	{
#ifdef ISCUDA
		return ::__logf(a);
#else
		return ::logf(a);
#endif
	}

	CUDA_FUNC_IN static float log2(float a)
	{
#ifdef ISCUDA
		return ::__log2f(a);
#else
		return ::logf(a) / ::logf(2.0f);
#endif
	}

	CUDA_FUNC_IN static float sin(float a)
	{
#ifdef ISCUDA
		return ::__sinf(a);
#else
		return ::sinf(a);
#endif
	}

	CUDA_FUNC_IN static float cos(float a)
	{
#ifdef ISCUDA
		return ::__cosf(a);
#else
		return ::cosf(a);
#endif
	}

	CUDA_FUNC_IN static float tan(float a)
	{
#ifdef ISCUDA
		return ::__tanf(a);
#else
		return ::tanf(a);
#endif
	}

	CUDA_FUNC_IN static unsigned int floatToBits(float a)
	{
#ifdef ISCUDA
		return ::__float_as_int(a);
#else
		return *(unsigned int*)&a;
#endif
	}

	CUDA_FUNC_IN static float bitsToFloat(unsigned int a)
	{
#ifdef ISCUDA
		return ::__int_as_float(a);
#else
		return *(float*)&a;
#endif
	}

	CUDA_FUNC_IN static float exp2(unsigned int a)
	{
#ifdef ISCUDA
		return ::exp2f((float)a); 
#else
		return bitsToFloat(clamp(a + 127, 1u, 254u) << 23);
#endif
	}

	CUDA_FUNC_IN static float fastMin(float a, float b)
	{
#ifdef ISCUDA
		return ::min(a, b);
#else
		return (a + b - abs(a - b)) * 0.5f;
#endif
	}

	CUDA_FUNC_IN static float fastMax(float a, float b)
	{
#ifdef ISCUDA
		return ::max(a, b);
#else
		return (a + b + abs(a - b)) * 0.5f;
#endif
	}

	CUDA_FUNC_IN static float    scale(float a, int b)  { return a * exp2((float)b); }
	CUDA_FUNC_IN static float    fastclamp(float v, float lo, float hi) { return fastMin(fastMax(v, lo), hi); }

	template <class T> CUDA_FUNC_IN static T sqr(const T& a) { return a * a; }
	template <class T> CUDA_FUNC_IN static T rcp(const T& a) { return (a) ? (T)1 / a : (T)0; }
	template <class A, class B> CUDA_FUNC_IN static A lerp(const A& a, const A& b, const B& t) { return (A)(a * ((B)1 - t) + b * t); }
};

