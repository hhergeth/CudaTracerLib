#pragma once

#include <Defines.h>
#include "Vector.h"
#include <iostream>
#ifdef ISCUDA
#include "cuda_fp16.h"
#endif

namespace CudaTracerLib {

struct half
{
	unsigned short val;
public:
	CUDA_FUNC_IN half()
	{

	}

	CUDA_FUNC_IN half(float f)
	{
#ifdef ISCUDA
		val = __float2half_rn(f);
#else
		/// https://devtalk.nvidia.com/default/topic/883897/cuda-programming-and-performance/error-when-trying-to-use-half-fp16-/post/4691963/#4691963
		uint32_t ia;
		memcpy(&ia, &f, sizeof(ia));
		uint16_t ir;

		ir = (ia >> 16) & 0x8000;
		if ((ia & 0x7f800000) == 0x7f800000) {
			if ((ia & 0x7fffffff) == 0x7f800000) {
				ir |= 0x7c00; /* infinity */
			}
			else {
				ir = 0x7fff; /* canonical NaN */
			}
		}
		else if ((ia & 0x7f800000) >= 0x33000000) {
			int shift = (int)((ia >> 23) & 0xff) - 127;
			if (shift > 15) {
				ir |= 0x7c00; /* infinity */
			}
			else {
				ia = (ia & 0x007fffff) | 0x00800000; /* extract mantissa */
				if (shift < -14) { /* denormal */
					ir |= ia >> (-1 - shift);
					ia = ia << (32 - (-1 - shift));
				}
				else { /* normal */
					ir |= ia >> (24 - 11);
					ia = ia << (32 - (24 - 11));
					ir = ir + ((14 + shift) << 10);
				}
				/* IEEE-754 round to nearest of even */
				if ((ia > 0x80000000) || ((ia == 0x80000000) && (ir & 1))) {
					ir++;
				}
			}
		}
		val = ir;
#endif
	}

	CUDA_FUNC_IN half(unsigned short s)
		: val(s)
	{

	}

	CUDA_FUNC_IN float ToFloat() const
	{
#ifdef ISCUDA
		return __half2float(val);
#else
		int fltInt32 = ((val & 0x8000) << 16);
		fltInt32 |= ((val & 0x7fff) << 13) + 0x38000000;

		float fRet;
		memcpy(&fRet, &fltInt32, sizeof(float));
		return fRet;
#endif
	}

	CUDA_FUNC_IN operator float() const
	{
		return ToFloat();
	}

	CUDA_FUNC_IN unsigned int bits() const
	{
		return val;
	}
};

struct half2
{
	half x, y;
	CUDA_FUNC_IN half2() {}
	CUDA_FUNC_IN half2(const Vec2f& v)
	{
		x = half(v.x);
		y = half(v.y);
	}
	CUDA_FUNC_IN half2(float _x, float _y)
	{
		x = half(_x);
		y = half(_y);
	}
	CUDA_FUNC_IN Vec2f ToFloat2() const
	{
		return Vec2f(x.ToFloat(), y.ToFloat());
	}
};

struct half3
{
	half x, y, z;
	CUDA_FUNC_IN half3() {}
	CUDA_FUNC_IN half3(const Vec3f& v)
	{
		x = half(v.x);
		y = half(v.y);
		z = half(v.z);
	}
	CUDA_FUNC_IN half3(float _x, float _y, float _z)
	{
		x = half(_x);
		y = half(_y);
		z = half(_z);
	}
	CUDA_FUNC_IN Vec3f ToFloat3() const
	{
		return Vec3f(x.ToFloat(), y.ToFloat(), z.ToFloat());
	}
};

struct half4
{
	half x, y, z, w;
	CUDA_FUNC_IN half4() {}
	CUDA_FUNC_IN half4(const Vec4f& v)
	{
		x = half(v.x);
		y = half(v.y);
		z = half(v.z);
		w = half(v.w);
	}
	CUDA_FUNC_IN half4(float _x, float _y, float _z, float _w)
	{
		x = half(_x);
		y = half(_y);
		z = half(_z);
		w = half(_w);
	}
	CUDA_FUNC_IN Vec4f ToFloat4() const
	{
		return Vec4f(x.ToFloat(), y.ToFloat(), z.ToFloat(), w.ToFloat());
	}
};

}