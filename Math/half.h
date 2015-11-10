#pragma once

#include "..\Defines.h"
#include "Vector.h"

///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002, Industrial Light & Magic, a division of Lucas
// Digital Ltd. LLC
// 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of Industrial Light & Magic nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission. 
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

// Primary authors:
//     Florian Kainz <kainz@ilm.com>
//     Rod Bogart <rgb@ilm.com>

//---------------------------------------------------------------------------
//
//	half -- a 16-bit floating point number class:
//
//	Type half can represent positive and negative numbers whose
//	magnitude is between roughly 6.1e-5 and 6.5e+4 with a relative
//	error of 9.8e-4; numbers smaller than 6.1e-5 can be represented
//	with an absolute error of 6.0e-8.  All integers from -2048 to
//	+2048 can be represented exactly.
//
//	Type half behaves (almost) like the built-in C++ floating point
//	types.  In arithmetic expressions, half, float and double can be
//	mixed freely.  Here are a few examples:
//
//	    half a (3.5);
//	    float b (a + sqrt (a));
//	    a += b;
//	    b += a;
//	    b = a + 7;
//
//	Conversions from half to float are lossless; all half numbers
//	are exactly representable as floats.
//
//	Conversions from float to half may not preserve a float's value
//	exactly.  If a float is not representable as a half, then the
//	float value is rounded to the nearest representable half.  If a
//	float value is exactly in the middle between the two closest
//	representable half values, then the float value is rounded to
//	the closest half whose least significant bit is zero.
//
//	Overflows during float-to-half conversions cause arithmetic
//	exceptions.  An overflow occurs when the float value to be
//	converted is too large to be represented as a half, or if the
//	float value is an infinity or a NAN.
//
//	The implementation of type half makes the following assumptions
//	about the implementation of the built-in C++ types:
//
//	    float is an IEEE 754 single-precision number
//	    sizeof (float) == 4
//	    sizeof (unsigned int) == sizeof (float)
//	    alignof (unsigned int) == alignof (float)
//	    sizeof (unsigned short) == 2
//
//---------------------------------------------------------------------------

class s10e5
{
 public:

  // constructors
  CUDA_FUNC_IN s10e5 ();			// no initialization
  CUDA_FUNC_IN s10e5 (float f);
  CUDA_FUNC_IN s10e5 (unsigned short s);
  CUDA_FUNC_IN float ToFloat() const
  {
	  return (float)*this;
  }

  // convert-out
  CUDA_FUNC_IN operator		float () const;

  // unary operators
  CUDA_FUNC_IN s10e5		operator - () const;

  // assign and op-assign

  CUDA_FUNC_IN s10e5 &		operator = (s10e5  h);
  CUDA_FUNC_IN s10e5 &		operator = (float f);

  CUDA_FUNC_IN s10e5 &		operator += (s10e5  h);
  CUDA_FUNC_IN s10e5 &		operator += (float f);

  CUDA_FUNC_IN s10e5 &		operator -= (s10e5  h);
  CUDA_FUNC_IN s10e5 &		operator -= (float f);

  CUDA_FUNC_IN s10e5 &		operator *= (s10e5  h);
  CUDA_FUNC_IN s10e5 &		operator *= (float f);

  CUDA_FUNC_IN s10e5 &		operator /= (s10e5  h);
  CUDA_FUNC_IN s10e5 &		operator /= (float f);


  //---------------------------------------------------------
  // Round to n-bit precision (n should be between 0 and 10).
  // After rounding, the significand's 10-n least significant
  // bits will be zero.
  //---------------------------------------------------------

  CUDA_FUNC_IN s10e5		round(unsigned int n) const;

  // ------------------------------ predicates ------------------------------
  // Use these with the syntax "if (x.isZero()) { ... }"
  CUDA_FUNC_IN bool	isFinite() const;       // true iff normal, subnormal or zero
  CUDA_FUNC_IN bool	isNormalized() const;	 // true iff normal
  CUDA_FUNC_IN bool	isDenormalized() const; // true iff subnormal
  CUDA_FUNC_IN bool	isZero() const;         // true iff zero or negative-zero
  CUDA_FUNC_IN bool	isNan() const;          // true iff NAN
  CUDA_FUNC_IN bool	isInfinity() const;     // true iff infinite
  CUDA_FUNC_IN bool	isNegative() const;     // true iff negative (includes negative NANs)

  // ---------------------------- special values ----------------------------
  CUDA_FUNC_IN static s10e5		posInf(); // returns +infinity
  CUDA_FUNC_IN static s10e5		negInf(); // returns -infinity
  CUDA_FUNC_IN static s10e5		qNan();   // returns a quiet NAN (0.11111.1111111111)
  CUDA_FUNC_IN static s10e5		Indet();  // "indeterminate" NAN (0.11111.1000000000)
  CUDA_FUNC_IN static s10e5		sNan();   // signaling NAN (0.11111.0111111111)

  // --------------------- access to raw representation ---------------------
  CUDA_FUNC_IN unsigned short	bits() const;
  CUDA_FUNC_IN void			setBits(unsigned short bits);

 public:
  // This union gives us an easy way to examine and/or set the individual
  // bit-fields of an s23e8.
  union u_u32_s23e8 {
	unsigned int i;
	float	 f;
  };

 private:
  unsigned short	_h;
};


// std::ostream &		operator << (std::ostream &os, s10e5  h);
// std::istream &		operator >> (std::istream &is, s10e5 &h);


#define S10E5_min	5.96046448e-08	// Smallest positive s10e5

#define S10E5_NRM_min	6.10351562e-05	// Smallest positive normalized s10e5

#define S10E5_max	65504.0		// Largest positive s10e5

#define S10E5_EPSILON	0.00097656	// Smallest positive e for which
					// s10e5 (1.0 + e) != s10e5 (1.0)

#define S10E5_MANT_DIG	11		// Number of digits in mantissa
					// (significand + hidden leading 1)

#define S10E5_DIG	2		// Number of base 10 digits that
					// can be represented without change

#define S10E5_RADIX	2		// Base of the exponent

#define S10E5_min_EXP	-13		// Minimum negative integer such that
					// S10E5_RADIX raised to the power of
					// one less than that integer is a
					// normalized s10e5

#define S10E5_max_EXP	16		// Maximum positive integer such that
					// S10E5_RADIX raised to the power of
					// one less than that integer is a
					// normalized s10e5

#define S10E5_min_10_EXP	-4		// Minimum positive integer such
					// that 10 raised to that power is
					// a normalized s10e5

#define S10E5_max_10_EXP	4		// Maximum positive integer such
					// that 10 raised to that power is
					// a normalized s10e5


//---------------------------------------------------------------------------
//
// Implementation --
//
// Representation of a float:
//
//	We assume that a float, f, is an IEEE 754 single-precision
//	floating point number, whose bits are arranged as follows:
//
//	    31 (msb)
//	    | 
//	    | 30     23
//	    | |      | 
//	    | |      | 22                    0 (lsb)
//	    | |      | |                     |
//	    X XXXXXXXX XXXXXXXXXXXXXXXXXXXXXXX
//
//	    s e        m
//
//	S is the sign-bit, e is the exponent and m is the significand.
//
//	If e is between 1 and 254, f is a normalized number:
//
//	            s    e-127
//	    f = (-1)  * 2      * 1.m
//
//	If e is 0, and m is not zero, f is a denormalized number:
//
//	            s    -126
//	    f = (-1)  * 2      * 0.m
//
//	If e and m are both zero, f is zero:
//
//	    f = 0.0
//
//	If e is 255, f is an "infinity" or "not a number" (NAN),
//	depending on whether m is zero or not.
//
//	Examples:
//
//	    0 00000000 00000000000000000000000 = 0.0
//	    0 01111110 00000000000000000000000 = 0.5
//	    0 01111111 00000000000000000000000 = 1.0
//	    0 10000000 00000000000000000000000 = 2.0
//	    0 10000000 10000000000000000000000 = 3.0
//	    1 10000101 11110000010000000000000 = -124.0625
//	    0 11111111 00000000000000000000000 = +infinity
//	    1 11111111 00000000000000000000000 = -infinity
//	    0 11111111 10000000000000000000000 = NAN
//	    1 11111111 11111111111111111111111 = NAN
//
// Representation of a s10e5:
//
//	Here is the bit-layout for a s10e5 number, h:
//
//	    15 (msb)
//	    | 
//	    | 14  10
//	    | |   |
//	    | |   | 9        0 (lsb)
//	    | |   | |        |
//	    X XXXXX XXXXXXXXXX
//
//	    s e     m
//
//	S is the sign-bit, e is the exponent and m is the significand.
//
//	If e is between 1 and 30, h is a normalized number:
//
//	            s    e-15
//	    h = (-1)  * 2     * 1.m
//
//	If e is 0, and m is not zero, h is a denormalized number:
//
//	            S    -14
//	    h = (-1)  * 2     * 0.m
//
//	If e and m are both zero, h is zero:
//
//	    h = 0.0
//
//	If e is 31, h is an "infinity" or "not a number" (NAN),
//	depending on whether m is zero or not.
//
//	Examples:
//
//	    0 00000 0000000000 = 0.0
//	    0 01110 0000000000 = 0.5
//	    0 01111 0000000000 = 1.0
//	    0 10000 0000000000 = 2.0
//	    0 10000 1000000000 = 3.0
//	    1 10101 1111000001 = -124.0625
//	    0 11111 0000000000 = +infinity
//	    1 11111 0000000000 = -infinity
//	    0 11111 1000000000 = NAN
//	    1 11111 1111111111 = NAN
//
// Conversion:
//
//	Converting from a float to a s10e5 requires some non-trivial bit
//	manipulations.  In some cases, this makes conversion relatively
//	slow, but the most common case is accelerated via table lookups.
//
//	Converting back from a s10e5 to a float is easier because we don't
//	have to do any rounding.  In addition, there are only 65536
//	different s10e5 numbers; we can convert each of those numbers once
//	and store the results in a table.  Later, all conversions can be
//	done using only simple table lookups.
//
//---------------------------------------------------------------------------


inline s10e5::s10e5 () { } // no initialization

// -------------------------- in-convert from s23e8 -------------------------

inline s10e5::s10e5 (float f)
{
  u_u32_s23e8 x;

  x.f = f;

  register int e = (x.i >> 23) & 0x000000ff;
  register int s = (x.i >> 16) & 0x00008000;
  register int m = x.i & 0x007fffff;

  e = e - 127;
  if (e == 128) {
	// infinity or NAN; preserve the leading bits of mantissa
	// because they tell whether it's a signaling of quiet NAN
	_h = s | (31 << 10) | (m >> 13);
  } else if (e > 15) {
	// overflow to infinity
	_h = s | (31 << 10);
  } else if (e > -15) {
	// normalized case
	if ((m & 0x00003fff) == 0x00001000) {
	  // tie, round down to even
	  _h = s | ((e+15) << 10) | (m >> 13);
	} else {
	  // all non-ties, and tie round up to even
	  _h = s | ((e+15) << 10) | ((m + 0x00001000) >> 13);
	}
  } else if (e > -25) {
	// convert to subnormal
	m |= 0x00800000; // restore the implied bit
	e = -14 - e; // shift count
	m >>= e; // M now in position but 2^13 too big
	if ((m & 0x00003fff) == 0x00001000) {
	  // tie round down to even
	} else {
	  // all non-ties, and tie round up to even
	  m += (1 << 12); // m += 0x00001000
	}
	m >>= 13;
	_h = s | m;
  } else {
	// zero, or underflow
	_h = s;
  }
}

inline s10e5::s10e5 (unsigned short s)
{
	_h = s;
}


// ------------------------ out-convert s10e5 to s23e8 ----------------------

inline s10e5::operator float () const
{
#ifdef ISCUDA
	return __half2float(_h);
#else
  register int s = _h & 0x8000;
  register int e = (_h & 0x7c00) >> 10;
  register int m = _h & 0x03ff;
  u_u32_s23e8 x;

  s <<= 16;
  if (e == 31) {
	// infinity or NAN
	e = 255 << 23;
	m <<= 13;
	x.i = s | e | m;
  } else if (e > 0) {
	// normalized
	e = e + (127 - 15);
	e <<= 23;
	m <<= 13;
	x.i = s | e | m;
  } else if (m == 0) {
	// zero
	x.i = s;
  } else {
	// subnormal, value is m times 2^-24
	x.f = ((float) m);
	x.i = s | (x.i - (24 << 23));
  }
  return(x.f);
#endif
}


//-------------------------
// Round to n-bit precision
//
// %%% this routine does not handle subnormals properly
//-------------------------

inline s10e5 s10e5::round (unsigned int n) const
{
  //
  // Parameter check.
  //

  if (n >= 10)
	return *this;

  //
  // Disassemble h into the sign, s,
  // and the combined exponent and significand, e.
  //

  unsigned short s = _h & 0x8000;
  unsigned short e = _h & 0x7fff;

  //
  // Round the exponent and significand to the nearest value
  // where ones occur only in the (10-n) most significant bits.
  // Note that the exponent adjusts automatically if rounding
  // up causes the significand to overflow.
  //

  e >>= 9 - n;
  e  += e & 1;
  e <<= 9 - n;

  //
  // Check for exponent overflow.
  //

  if (e >= 0x7c00)
	{
	  //
	  // Overflow occurred -- truncate instead of rounding.
	  //

	  e = _h;
	  e >>= 10 - n;
	  e <<= 10 - n;
	}

  //
  // Put the original sign bit back.
  //

  s10e5 h;
  h._h = s | e;

  return h;
}


//-----------------------
// Other inline functions
//-----------------------

// ------------------------------- assignment -------------------------------

inline s10e5 & s10e5::operator = (s10e5 h)
{ _h = h._h; return *this; }

inline s10e5 & s10e5::operator = (float f)
{ *this = s10e5 (f); return *this; }

// ---------------------------- unary operators -----------------------------

inline s10e5 s10e5::operator - () const
{ s10e5 h; h._h = _h ^ 0x8000; return h; }

// ---------------------- assign with binary operator -----------------------

inline s10e5 & s10e5::operator += (s10e5 h)
{ *this = s10e5 (float (*this) + float (h)); return *this; }

inline s10e5 & s10e5::operator += (float f)
{ *this = s10e5 (float (*this) + f); return *this; }

inline s10e5 & s10e5::operator -= (s10e5 h)
{ *this = s10e5 (float (*this) - float (h)); return *this; }

inline s10e5 & s10e5::operator -= (float f)
{ *this = s10e5 (float (*this) - f); return *this; }

inline s10e5 & s10e5::operator *= (s10e5 h)
{ *this = s10e5 (float (*this) * float (h)); return *this; }

inline s10e5 & s10e5::operator *= (float f)
{ *this = s10e5 (float (*this) * f); return *this; }

inline s10e5 & s10e5::operator /= (s10e5 h)
{ *this = s10e5 (float (*this) / float (h)); return *this; }

inline s10e5 & s10e5::operator /= (float f)
{ *this = s10e5 (float (*this) / f); return *this; }

// ------------------------------- predicates -------------------------------

inline bool s10e5::isFinite () const
{ unsigned short e = (_h >> 10) & 0x001f; return e < 31; }

inline bool s10e5::isNormalized () const
{ unsigned short e = (_h >> 10) & 0x001f; return e > 0 && e < 31; }

inline bool s10e5::isDenormalized () const
{
  unsigned short e = (_h >> 10) & 0x001f;
  unsigned short m =  _h & 0x3ff;
  return e == 0 && m != 0;
}

inline bool s10e5::isZero () const
{ return (_h & 0x7fff) == 0; }

inline bool s10e5::isNan () const
{
  unsigned short e = (_h >> 10) & 0x001f;
  unsigned short m =  _h & 0x3ff;
  return e == 31 && m != 0;
}

inline bool s10e5::isInfinity () const
{
  unsigned short e = (_h >> 10) & 0x001f;
  unsigned short m =  _h & 0x3ff;
  return e == 31 && m == 0;
}

inline bool s10e5::isNegative () const
{ return (_h & 0x8000) != 0; }

// ----------------------------- special values -----------------------------

inline s10e5 s10e5::posInf () { s10e5 h; h._h = 0x7c00; return h; }
inline s10e5 s10e5::negInf () { s10e5 h; h._h = 0xfc00; return h; }
inline s10e5 s10e5::qNan () { s10e5 h; h._h = 0x7fff; return h; }
inline s10e5 s10e5::sNan () { s10e5 h; h._h = 0x7dff; return h; }
inline s10e5 s10e5::Indet () { s10e5 h; h._h = 0x7e00; return h; }

// ---------------------- access to raw representation ----------------------

inline unsigned short s10e5::bits () const { return _h; }
inline void s10e5::setBits (unsigned short bits) { _h = bits; }

typedef s10e5 half;

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
	CUDA_FUNC_IN Vec2f ToFloat2()
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
	CUDA_FUNC_IN Vec3f ToFloat3()
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
	CUDA_FUNC_IN Vec4f ToFloat4()
	{
		return Vec4f(x.ToFloat(), y.ToFloat(), z.ToFloat(), w.ToFloat());
	}
};