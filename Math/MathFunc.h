#pragma once

#include <Defines.h>
#include <math.h>
#include <float.h>
#pragma warning(push, 3)
#include <vector_functions.h> // float4, etc.
#pragma warning(pop)

namespace CudaTracerLib {

#define PI     3.14159265358979f
#define INV_PI (1.0f / PI)
#define INV_TWOPI (1.0f / (2.0f * PI))
#define INV_FOURPI (1.0f / (4.0f * PI))
#define SQRT_TWO      1.41421356237309504880f
#define INV_SQRT_TWO  0.70710678118654752440f
#define RCPOVERFLOW   2.93873587705571876e-39f

//general epsilon used everywhere none of the specific ones is applicable
#define EPSILON 0.000001f
//epsilon for checking the next ray triangle intersection
#define MIN_RAYTRACE_DISTANCE (1e-4f)
//epsilon used for comparing given directions to perfect specular/delta directions
#define DeltaEpsilon 1e-3f

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
	//credits to HAKMEM

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
	*a = sinf(f);
	*b = cosf(f);
}
#endif

//http://stackoverflow.com/a/31010352/1715849
CUDA_FUNC_IN int floatToOrderedInt(float floatVal)
{
	int intVal = float_as_int_(floatVal);
	return (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF;
}

CUDA_FUNC_IN float orderedIntToFloat(int intVal)
{
	return int_as_float_((intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF);
}

#define FW_SPECIALIZE_MINMAX(TEMPLATE, T, K, MIN, MAX) \
	TEMPLATE CUDA_FUNC_IN T min(K T& a, K T& b) { return MIN; } \
	TEMPLATE CUDA_FUNC_IN T max(K T& a, K T& b) { return MAX; } \
	TEMPLATE CUDA_FUNC_IN T min(K T& a, K T& b, K T& c) { T d = min(a, b); return min(d, c); } \
	TEMPLATE CUDA_FUNC_IN T max(K T& a, K T& b, K T& c) { T d = max(a, b); return max(d, c); } \
	TEMPLATE CUDA_FUNC_IN T min(K T& a, K T& b, K T& c, K T& d) { T e = min(a,b,c); return min(e, d); } \
	TEMPLATE CUDA_FUNC_IN T max(K T& a, K T& b, K T& c, K T& d) { T e = max(a,b,c); return max(e, d); } \

FW_SPECIALIZE_MINMAX(template <class T>, T, , (a < b) ? a : b, (a > b) ? a : b)
FW_SPECIALIZE_MINMAX(template <class T>, T, const, (a < b) ? a : b, (a > b) ? a : b)

class math
{
public:

	CUDA_FUNC_IN static float safe_asin(float value) {
		return asinf(min(1.0f, max(-1.0f, value)));
	}

	CUDA_FUNC_IN static float safe_acos(float value) {
		return acosf(min(1.0f, max(-1.0f, value)));
	}

	CUDA_FUNC_IN static float safe_sqrt(float value) {
		return sqrt(max(0.0f, value));
	}

	CUDA_FUNC_IN static float signum(float value)
	{
		return copysignf(1.0f, value);
	}

	CUDA_FUNC_IN static int modulo(int a, int b)
	{
		int r = a % b;
		return (r < 0) ? r + b : r;
	}

	CUDA_FUNC_IN static float modulo(float a, float b)
	{
		float r = fmodf(a, b);
		return (r < 0) ? r + b : r;
	}

	template<typename T> CUDA_FUNC_IN static T sign(T f)
	{
		return f > T(0) ? T(1) : (f < T(0) ? T(-1) : T(0));
	}

	CUDA_FUNC_IN static float frac(float f)
	{
		return f - floorf(f);
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

	template <class A, class B> CUDA_FUNC_IN static A lerp(const A& a, const A& b, const B& t) { return (A)(a * ((B)1 - t) + b * t); }

	template<typename T> CUDA_FUNC_IN static T bilerp2(const T& lt, const T& rt, const T& ld, const T& rd, const float2& uv)
	{
		T a = lt + (rt - lt) * uv.x, b = ld + (rd - ld) * uv.x;
		return a + (b - a) * uv.y;
	}

	template <class T> CUDA_FUNC_IN static T clamp(T& v, T& lo, T& hi) { return min(max(v, lo), hi); }
	template <class T> CUDA_FUNC_IN static T clamp(const T& v, const T& lo, const T& hi) { return min(max(v, lo), hi); }
	template <class T> CUDA_FUNC_IN static T clamp01(T& a) { return clamp(a, T(0.0f), T(1.0f)); }
	template <class T> CUDA_FUNC_IN static T clamp01(const T& a) { return clamp(a, T(0.0f), T(1.0f)); }

	CUDA_FUNC_IN static int Mod(int a, int b)
	{
		int n = int(a / b);
		a -= n*b;
		if (a < 0) a += b;
		return a;
	}

	CUDA_FUNC_IN static float Radians(float deg)
	{
		return ((float)PI / 180.f) * deg;
	}

	CUDA_FUNC_IN static float Degrees(float rad)
	{
		return (180.f / (float)PI) * rad;
	}

	CUDA_FUNC_IN static bool IsPowerOf2(int v)
	{
		return (v & (v - 1)) == 0;
	}

	CUDA_FUNC_IN static bool IsNaN(float value)
	{
		return value != value;
	}

	CUDA_FUNC_IN static unsigned int RoundUpPow2(unsigned int v)
	{
		//Credits to PBRT.
		v--;
		v |= v >> 1;    v |= v >> 2;
		v |= v >> 4;    v |= v >> 8;
		v |= v >> 16;
		return v + 1;
	}

	CUDA_FUNC_IN static float    sqrt(float a)         { return ::sqrtf(a); }
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

	CUDA_FUNC_IN static int Log2Int(float v)
	{
		return Floor2Int(log2(v));
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

	CUDA_FUNC_IN static float exp2(unsigned int a)
	{
#ifdef ISCUDA
		return ::exp2f((float)a);
#else
		return int_as_float_(clamp(a + 127, 1u, 254u) << 23);
#endif
	}

	CUDA_FUNC_IN static float fastMin(float a, float b)
	{
#ifdef ISCUDA
		return ::min(a, b);
#else
		return (a + b - math::abs(a - b)) * 0.5f;
#endif
	}

	CUDA_FUNC_IN static float fastMax(float a, float b)
	{
#ifdef ISCUDA
		return ::max(a, b);
#else
		return (a + b + math::abs(a - b)) * 0.5f;
#endif
	}

	CUDA_FUNC_IN static float hypot2(float a, float b)
	{
		float r;
		if (math::abs(a) > math::abs(b)) {
			r = b / a;
			r = math::abs(a) * math::sqrt(1.0f + r*r);
		}
		else if (b != 0.0f) {
			r = a / b;
			r = math::abs(b) * math::sqrt(1.0f + r*r);
		}
		else {
			r = 0.0f;
		}
		return r;
	}

	CUDA_FUNC_IN static float erfinv(float x)
	{
		// Based on "Approximating the erfinv function" by Mark Giles
		float w = -math::log((1.0f - x)*(1.0f + x));
		float p;
		if (w < 5.0f) {
			w = w - 2.5f;
			p =  2.81022636e-08f;
			p =  3.43273939e-07f + p*w;
			p = -3.5233877e-06f + p*w;
			p = -4.39150654e-06f + p*w;
			p =  0.00021858087f + p*w;
			p = -0.00125372503f + p*w;
			p = -0.00417768164f + p*w;
			p =  0.246640727f + p*w;
			p =  1.50140941f + p*w;
		}
		else {
			w = math::sqrt(w) - 3;
			p = -0.000200214257f;
			p =  0.000100950558f + p*w;
			p =  0.00134934322f + p*w;
			p = -0.00367342844f + p*w;
			p =  0.00573950773f + p*w;
			p = -0.0076224613f + p*w;
			p =  0.00943887047f + p*w;
			p =  1.00167406f + p*w;
			p =  2.83297682f + p*w;
		}
		return p*x;
	}

	CUDA_FUNC_IN float static erf(float x)
	{
		float a1 =   0.254829592f;
		float a2 = -0.284496736f;
		float a3 =   1.421413741f;
		float a4 = -1.453152027f;
		float a5 =   1.061405429f;
		float p =   0.3275911f;

		// Save the sign of x
		float sign = math::signum(x);
		x = math::abs(x);

		// A&S formula 7.1.26
		float t = 1.0f / (1.0f + p*x);
		float y = 1.0f - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*math::exp(-x*x);

		return sign*y;
	}

	CUDA_FUNC_IN static float    scale(float a, int b)  { return a * exp2((float)b); }
	CUDA_FUNC_IN static float    fastclamp(float v, float lo, float hi) { return fastMin(fastMax(v, lo), hi); }

	template <class T> CUDA_FUNC_IN static T sqr(const T& a) { return a * a; }
	template <class T> CUDA_FUNC_IN static T rcp(const T& a) { return (a) ? (T)1 / a : (T)0; }
};

class kepler_math
{
public:
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
};

template<int DIM> struct pow_int_compile
{
	template<typename T> CUDA_FUNC_IN static float pow(const T& f)
	{
		return f * pow_int_compile<DIM - 1>::pow(f);
	}
};

template<> struct pow_int_compile<0>
{
	template<typename T>  CUDA_FUNC_IN static float pow(const T& f)
	{
		return (T)1;
	}
};

}
