#pragma once

#include "..\Defines.h"
#include <math.h>
#include <cmath>
#include <float.h>
#include "emmintrin.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "vector_functions.h"
#include "cutil_math.h"
#include <Windows.h>

#define ISBLACK(v) (fsumf(v) == 0.0f)

#define BADFLOAT(x) ((*(uint*)&x & 0x7f000000) == 0x7f000000)
void*mymalloc(int s);
//not the right place for this kind of stuff
#define MALLOC(x,t) ((t*)mymalloc((x)*sizeof(t)))
#define FREE(x)(_aligned_free((x)))
#define CAST(x,t)(reinterpret_cast<t>(x))
#define PI     3.14159265358979f
#define INV_PI (1.0f / PI)
#define INV_TWOPI (1.0f / (2.0f * PI))
#define INV_FOURPI (1.0f / (4.0f * PI))
//#define PI_2 (2.0f * PI)

// The maximum possible value for a 32-bit floating point variable
#ifndef  MAXFLOAT
#define  MAXFLOAT   ((float)3.40282347e+38) 
#endif

#define MAXDEPTH 20

// When ray tracing, the epsilon that t > EPSILON in order to avoid self intersections
#define EPSILON       2e-5

#define FINDMINMAX( x0, x1, x2, min, max ) \
  min = max = x0; if(x1<min) min=x1; if(x1>max) max=x1; if(x2<min) min=x2; if(x2>max) max=x2;
// X-tests
#define AXISTEST_X01( a, b, fa, fb )											\
	p0 = a * v0.y - b * v0.z, p2 = a * v2.y - b * v2.z; \
    if (p0 < p2) { min = p0; max = p2;} else { min = p2; max = p0; }			\
	rad = fa * a_BoxHalfsize.y + fb * a_BoxHalfsize.z;				\
	if (min > rad || max < -rad) return 0;
#define AXISTEST_X2( a, b, fa, fb )												\
	p0 = a * v0.y - b * v0.z, p1 = a * v1.y - b * v1.z;	\
    if (p0 < p1) { min = p0; max = p1; } else { min = p1; max = p0;}			\
	rad = fa * a_BoxHalfsize.y + fb * a_BoxHalfsize.z;				\
	if(min>rad || max<-rad) return 0;
// Y-tests
#define AXISTEST_Y02( a, b, fa, fb )											\
	p0 = -a * v0.x + b * v0.z, p2 = -a * v2.x + b * v2.z; \
    if(p0 < p2) { min = p0; max = p2; } else { min = p2; max = p0; }			\
	rad = fa * a_BoxHalfsize.x + fb * a_BoxHalfsize.z;				\
	if (min > rad || max < -rad) return 0;
#define AXISTEST_Y1( a, b, fa, fb )												\
	p0 = -a * v0.x + b * v0.z, p1 = -a * v1.x + b * v1.z; \
    if (p0 < p1) { min = p0; max = p1; } else { min = p1; max = p0; }			\
	rad = fa * a_BoxHalfsize.x + fb * a_BoxHalfsize.z;				\
	if (min > rad || max < -rad) return 0;
// Z-tests
#define AXISTEST_Z12( a, b, fa, fb )											\
	p1 = a * v1.x - b * v1.y, p2 = a * v2.x - b * v2.y; \
    if(p2 < p1) { min = p2; max = p1; } else { min = p1; max = p2; }			\
	rad = fa * a_BoxHalfsize.x + fb * a_BoxHalfsize.y;				\
	if (min > rad || max < -rad) return 0;
#define AXISTEST_Z0( a, b, fa, fb )												\
	p0 = a * v0.x - b * v0.y, p1 = a * v1.x - b * v1.y;	\
    if(p0 < p1) { min = p0; max = p1; } else { min = p1; max = p0; }			\
	rad = fa * a_BoxHalfsize.x + fb * a_BoxHalfsize.y;				\
	if (min > rad || max < -rad) return 0;

class float4x4
	{
	public:
		union
		{
			struct { float4 Row[4]; };
			struct { float4 X, Y, Z, W; };
		};
		CUDA_FUNC_IN float4x4()
		{
		}
		static float4x4 NewIdentity()
		{
			float4x4 v;
			v.X = make_float4( 1, 0, 0, 0 );
			v.Y = make_float4( 0, 1, 0, 0 );
			v.Z = make_float4( 0, 0, 1, 0 );
			v.W = make_float4( 0, 0, 0, 1 );
			return v;
		}
		CUDA_FUNC_IN float4x4( float4 x, float4 y, float4 z, float4 w ) { X = x; Y = y; Z = z; W = w; }
		CUDA_FUNC_IN float4x4( float xx, float xy, float xz, float xw,
			float yx, float yy, float yz, float yw,
			float zx, float zy, float zz, float zw,
			float wx, float wy, float wz, float ww )
		{
			X = make_float4( xx, xy, xz, xw );
			Y = make_float4( yx, yy, yz, yw );
			Z = make_float4( zx, zy, zz, zw );
			W = make_float4( wx, wy, wz, ww );
		}

		CUDA_FUNC_IN float4  operator[] ( int i ) const	{ return Row[i]; }
		CUDA_FUNC_IN float4& operator[] ( int i ) 		{ return Row[i]; }

		CUDA_FUNC_IN void operator *= ( const float4x4& b )
		{
			float4x4 a = *this;
			*this = a * b;
		}

		CUDA_FUNC_IN float4x4 operator + ( const float4x4& a ) const
		{
			float4x4 b;
			b.X = this->X + a.X;
			b.Y = this->Y + a.Y;
			b.Z = this->Z + a.Z;
			b.W = this->W + a.W;
			return b;
		}

		CUDA_FUNC_IN float4x4 operator * ( const float4x4& a ) const
		{
			float4x4 b = *this;
			return float4x4(
				dot( b.X, make_float4( a.Row[0].x, a.Row[1].x, a.Row[2].x, a.Row[3].x ) ),
				dot( b.X, make_float4( a.Row[0].y, a.Row[1].y, a.Row[2].y, a.Row[3].y ) ),
				dot( b.X, make_float4( a.Row[0].z, a.Row[1].z, a.Row[2].z, a.Row[3].z ) ),
				dot( b.X, make_float4( a.Row[0].w, a.Row[1].w, a.Row[2].w, a.Row[3].w ) ),

				dot( b.Y, make_float4( a.Row[0].x, a.Row[1].x, a.Row[2].x, a.Row[3].x ) ),
				dot( b.Y, make_float4( a.Row[0].y, a.Row[1].y, a.Row[2].y, a.Row[3].y ) ),
				dot( b.Y, make_float4( a.Row[0].z, a.Row[1].z, a.Row[2].z, a.Row[3].z ) ),
				dot( b.Y, make_float4( a.Row[0].w, a.Row[1].w, a.Row[2].w, a.Row[3].w ) ),

				dot( b.Z, make_float4( a.Row[0].x, a.Row[1].x, a.Row[2].x, a.Row[3].x ) ),
				dot( b.Z, make_float4( a.Row[0].y, a.Row[1].y, a.Row[2].y, a.Row[3].y ) ),
				dot( b.Z, make_float4( a.Row[0].z, a.Row[1].z, a.Row[2].z, a.Row[3].z ) ),
				dot( b.Z, make_float4( a.Row[0].w, a.Row[1].w, a.Row[2].w, a.Row[3].w ) ),

				dot( b.W, make_float4( a.Row[0].x, a.Row[1].x, a.Row[2].x, a.Row[3].x ) ),
				dot( b.W, make_float4( a.Row[0].y, a.Row[1].y, a.Row[2].y, a.Row[3].y ) ),
				dot( b.W, make_float4( a.Row[0].z, a.Row[1].z, a.Row[2].z, a.Row[3].z ) ),
				dot( b.W, make_float4( a.Row[0].w, a.Row[1].w, a.Row[2].w, a.Row[3].w ) )
				);
		}
		CUDA_FUNC_IN float4 operator * ( const float4& a ) const
		{
			return make_float4(
				dot( a, make_float4( Row[0].x, Row[1].x, Row[2].x, Row[3].x ) ),
				dot( a, make_float4( Row[0].y, Row[1].y, Row[2].y, Row[3].y ) ),
				dot( a, make_float4( Row[0].z, Row[1].z, Row[2].z, Row[3].z ) ),
				dot( a, make_float4( Row[0].w, Row[1].w, Row[2].w, Row[3].w ) )
				);
		}
		CUDA_FUNC_IN float3 operator * ( const float3& b ) const
		{
			float4 a = make_float4(b.x, b.y, b.z, 1.0f);
			float4 r = make_float4(
				dot( a, make_float4( Row[0].x, Row[1].x, Row[2].x, Row[3].x ) ),
				dot( a, make_float4( Row[0].y, Row[1].y, Row[2].y, Row[3].y ) ),
				dot( a, make_float4( Row[0].z, Row[1].z, Row[2].z, Row[3].z ) ),
				dot( a, make_float4( Row[0].w, Row[1].w, Row[2].w, Row[3].w ) )
				);
			return make_float3(r.x, r.y, r.z) / r.w;
		}

		CUDA_FUNC_IN float3 TransformNormal(const float3& b) const
		{
			float4 a = make_float4(b.x, b.y, b.z, 0.0f);
			float4 r = make_float4(
				dot( a, make_float4( Row[0].x, Row[1].x, Row[2].x, Row[3].x ) ),
				dot( a, make_float4( Row[0].y, Row[1].y, Row[2].y, Row[3].y ) ),
				dot( a, make_float4( Row[0].z, Row[1].z, Row[2].z, Row[3].z ) ),
				dot( a, make_float4( Row[0].w, Row[1].w, Row[2].w, Row[3].w ) )
				);
			return make_float3(r.x, r.y, r.z);
		}

		CUDA_FUNC_IN float4x4 operator * ( const float a ) const
		{
			float4x4 b;
			b = *this;
			b.X *= a;
			b.Y *= a;
			b.Z *= a;
			b.W *= a;
			return b;
		}
		
		static float4x4 LookAt(const float3 _From, const float3 _To, const float3 _Up)
		{
			float3 forward = normalize(_To-_From);
			float3 side = normalize(cross(forward,_Up));
			float3 up = normalize(cross(side,forward));
			float4x4 mat=float4x4::Identity();
#define B(a,b) a = make_float4(b, a.w)
			B(mat.X, side);
			B(mat.Y, up);
			B(mat.Z, forward);
			B(mat.W, _From);
#undef B
			return mat;
		}

		static float4x4 Perspective(const float fov, const float asp, const float n, const float f)
		{
			float cosfov = cosf(0.5f * fov), sinfov = sinf(0.5f * fov), h = cosfov / sinfov, w = h / asp;
			float4x4 mat=float4x4::Identity();
			mat.X = make_float4(w,0,0,0);
			mat.Y = make_float4(0,h,0,0);
			mat.Z = make_float4(0,0,f / (f - n),1);
			mat.W = make_float4(0,0,- (n * f) / (f - n),0);
			return mat;
		}

		static float4x4 Lerp(const float4x4&a, const float4x4&b, float t)
		{
			float t2 = 1.0f - t;
			float4x4 r;
			for(int i = 0; i < 4; i++)
				r.Row[i] = a.Row[i] * t2 + b.Row[i] * t;
			return r;
		}

		static float4x4 RotateX( float a )
		{
			float x = cosf( a );
			float y = sinf( a );
			return float4x4(1, 0, 0, 0,	0, x, y, 0,	0, -y, x, 0, 0, 0, 0, 1 );
		}
		static float4x4 RotateY( float a )
		{
			float x = cosf( a );
			float y = sinf( a );
			return float4x4( x, 0, -y, 0, 0, 1, 0, 0, y, 0, x, 0, 0, 0, 0, 1 );
		}
		static float4x4 RotateZ( float a )
		{
			float x = cosf( a );
			float y = sinf( a );
			return float4x4( x, y, 0, 0, -y, x, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 );
		}
		static float4x4 Rotate( float3 _XYZ )
		{
			return RotateX(_XYZ.x)*RotateY(_XYZ.y)*RotateZ(_XYZ.z);
		}
		static float4x4 RotationAxis(const float3& _axis, const float angle)
		{
			float3 axis = _axis;
			float4x4 t = float4x4::Identity();
			float num = axis.x;
			float num2 = axis.y;
			float num3 = axis.z;
			float expr_20 = num2;
			float arg_26_0 = expr_20 * expr_20;
			float expr_24 = num;
			float arg_2B_0 = (arg_26_0 + expr_24 * expr_24);
			float expr_29 = num3;
			if ((arg_2B_0 + expr_29 * expr_29) != 1.0f)
				axis = normalize(axis);
			float x = axis.x;
			float y = axis.y;
			float z = axis.z;
			float num4 = cosf(angle);
			float num5 = sinf(angle);
			float num6 = x;
			float expr_74 = num6;
			float num7 = expr_74 * expr_74;
			float num8 = y;
			float expr_7F = num8;
			float num9 = expr_7F * expr_7F;
			float num10 = z;
			float expr_8A = num10;
			float num11 = (expr_8A * expr_8A);
			float num12 = (y * x);
			float num13 = (z * x);
			float num14 = (z * y);
			float num15 = num12;
			float num16 = num15 - num4 * num15;
			float num17 = num5 * z;
			float num18 = num13;
			float num19 = num18 - num4 * num18;
			float num20 = num5 * y;
			t[0] = make_float4((1.0f - num7) * num4 + num7, num17 + num16, num19 - num20,0);
			float num21 = num14;
			float num22 = num21 - num4 * num21;
			float num23 = num5 * x;
			t[1] = make_float4(num16 - num17, (1.0f - num9) * num4 + num9, num23 + num22,0);
			t[2] = make_float4(num20 + num19, num22 - num23, (1.0f - num11) * num4 + num11,0);
			return t;
		}

		CUDA_FUNC_IN float4x4 Transpose()
		{
			return float4x4( X.x, Y.x, Z.x, W.x, X.y, Y.y, Z.y, W.y, X.z, Y.z, Z.z, W.z, X.w, Y.w, Z.w, W.w);
		}

		CUDA_FUNC_IN float4x4 TransposeOpenGL()
		{
			return float4x4(X.x, X.y, X.z, W.x, 
							Y.x, Y.y, Y.z, W.y, 
							Z.x, Z.y, Z.z, W.z, 
							X.w, Y.w, Z.w, W.w);
		}

		static const float4x4 Identity()
		{
			return float4x4( 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
		}
		
		static float4x4 Translate( float3 t )
		{
			return float4x4( 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, t.x, t.y, t.z, 1 );
		}
		
		static float4x4 Translate( float x, float y, float z )
		{
			return float4x4( 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1 );
		}

		CUDA_FUNC_IN void OrthoNormalize()
		{
#define A(a) make_float3(a.x, a.y, a.z)
#define B(a,b) a = make_float4(b, a.w)
			float4x4 n;
			B(n.X, normalize( cross( A(Y), A(Z) ) ) );
			B(n.Y, normalize( cross( A(Z), A(X) ) ) );
			B(n.Z, normalize( cross( A(X), A(Y) ) ) );
			n.W = W;
			*this = n;
#undef A
#undef B
		}

		static float4x4 Scale( float3 s )
		{
			return float4x4( s.x, 0, 0, 0, 0, s.y, 0, 0, 0, 0, s.z, 0, 0, 0, 0, 1 );
		}
		static float4x4 Scale( float s )
		{
			return float4x4( s, 0, 0, 0, 0, s, 0, 0, 0, 0, s, 0, 0, 0, 0, 1 );
		}

		CUDA_FUNC_IN float4x4 Inverse() const
		{
			float m00 = Row[0].x, m01 = Row[0].y, m02 = Row[0].z, m03 = Row[0].w;
			float m10 = Row[1].x, m11 = Row[1].y, m12 = Row[1].z, m13 = Row[1].w;
			float m20 = Row[2].x, m21 = Row[2].y, m22 = Row[2].z, m23 = Row[2].w;
			float m30 = Row[3].x, m31 = Row[3].y, m32 = Row[3].z, m33 = Row[3].w;

			float v0 = m20 * m31 - m21 * m30;
			float v1 = m20 * m32 - m22 * m30;
			float v2 = m20 * m33 - m23 * m30;
			float v3 = m21 * m32 - m22 * m31;
			float v4 = m21 * m33 - m23 * m31;
			float v5 = m22 * m33 - m23 * m32;

			float t00 = + (v5 * m11 - v4 * m12 + v3 * m13);
			float t10 = - (v5 * m10 - v2 * m12 + v1 * m13);
			float t20 = + (v4 * m10 - v2 * m11 + v0 * m13);
			float t30 = - (v3 * m10 - v1 * m11 + v0 * m12);

			float invDet = 1 / (t00 * m00 + t10 * m01 + t20 * m02 + t30 * m03);

			float d00 = t00 * invDet;
			float d10 = t10 * invDet;
			float d20 = t20 * invDet;
			float d30 = t30 * invDet;

			float d01 = - (v5 * m01 - v4 * m02 + v3 * m03) * invDet;
			float d11 = + (v5 * m00 - v2 * m02 + v1 * m03) * invDet;
			float d21 = - (v4 * m00 - v2 * m01 + v0 * m03) * invDet;
			float d31 = + (v3 * m00 - v1 * m01 + v0 * m02) * invDet;

			v0 = m10 * m31 - m11 * m30;
			v1 = m10 * m32 - m12 * m30;
			v2 = m10 * m33 - m13 * m30;
			v3 = m11 * m32 - m12 * m31;
			v4 = m11 * m33 - m13 * m31;
			v5 = m12 * m33 - m13 * m32;

			float d02 = + (v5 * m01 - v4 * m02 + v3 * m03) * invDet;
			float d12 = - (v5 * m00 - v2 * m02 + v1 * m03) * invDet;
			float d22 = + (v4 * m00 - v2 * m01 + v0 * m03) * invDet;
			float d32 = - (v3 * m00 - v1 * m01 + v0 * m02) * invDet;

			v0 = m21 * m10 - m20 * m11;
			v1 = m22 * m10 - m20 * m12;
			v2 = m23 * m10 - m20 * m13;
			v3 = m22 * m11 - m21 * m12;
			v4 = m23 * m11 - m21 * m13;
			v5 = m23 * m12 - m22 * m13;

			float d03 = - (v5 * m01 - v4 * m02 + v3 * m03) * invDet;
			float d13 = + (v5 * m00 - v2 * m02 + v1 * m03) * invDet;
			float d23 = - (v4 * m00 - v2 * m01 + v0 * m03) * invDet;
			float d33 = + (v3 * m00 - v1 * m01 + v0 * m02) * invDet;

			return float4x4(
				d00, d01, d02, d03,
				d10, d11, d12, d13,
				d20, d21, d22, d23,
				d30, d31, d32, d33);
		}

		CUDA_FUNC_IN float3 Translation()
		{
			return make_float3(W.x, W.y, W.z);
		}
		CUDA_FUNC_IN void Translation(float3 t)
		{
			W.x = t.x;
			W.y = t.y;
			W.z = t.z;
		}

		CUDA_FUNC_IN float3 Scale()
		{
			return make_float3(length(!X), length(!Y), length(!Z));
		}

		CUDA_FUNC_IN float3 Forward()
		{
			return make_float3(Z.x, Z.y, Z.z);
		}
		CUDA_FUNC_IN float3 Right()
		{
			return make_float3(X.x, X.y, X.z);
		}
		CUDA_FUNC_IN float3 Up()
		{
			return make_float3(Y.x, Y.y, Y.z);
		}
	};

class Quaternion
{
public:
	float4 val;
	inline float& operator[](int n) { return *(((float*)&val) + n); }
	inline float operator[](int n) const { return *(((float*)&val) + n); }
	CUDA_FUNC_IN Quaternion(){}
	CUDA_FUNC_IN Quaternion(float x, float y, float z, float w)
	{
		val = make_float4(x,y,z,w);
	}
	CUDA_FUNC_IN Quaternion(float x, float y, float z)
	{
		float w = 1.0f - x*x - y*y - z*z;
		w = w < 0.0 ? 0.0f : (float)-sqrt( double(w) );
		val = make_float4(x,y,z,w);
		normalize();
	}
	CUDA_FUNC_IN Quaternion operator *(const Quaternion &q) const
	{
		Quaternion r;
		r.val.w = val.w*q.val.w - val.x*q.val.x - val.y*q.val.y - val.z*q.val.z;
		r.val.x = val.w*q.val.x + val.x*q.val.w + val.y*q.val.z - val.z*q.val.y;
		r.val.y = val.w*q.val.y + val.y*q.val.w + val.z*q.val.x - val.x*q.val.z;
		r.val.z = val.w*q.val.z + val.z*q.val.w + val.x*q.val.y - val.y*q.val.x;
		return r;
	}
	CUDA_FUNC_IN float3 operator *(const float3 &v) const
	{
        float x = val.x + val.x;
        float y = val.y + val.y;
        float z = val.z + val.z;
        float wx = val.w * x;
        float wy = val.w * y;
        float wz = val.w * z;
        float xx = val.x * x;
        float xy = val.x * y;
        float xz = val.x * z;
        float yy = val.y * y;
        float yz = val.y * z;
        float zz = val.z * z;
		float3 vector;
        vector.x = ((v.x * ((1.0f - yy) - zz)) + (v.y * (xy - wz))) + (v.z * (xz + wy));
		vector.y = ((v.x * (xy + wz)) + (v.y * ((1.0f - xx) - zz))) + (v.z * (yz - wx));
        vector.z = ((v.x * (xz - wy)) + (v.y * (yz + wx))) + (v.z * ((1.0f - xx) - yy));
        return vector;
	}
	CUDA_FUNC_IN const Quaternion & operator *= (const Quaternion &q)
	{
		val.w = val.w*q.val.w - val.x*q.val.x - val.y*q.val.y - val.z*q.val.z;
		val.x = val.w*q.val.x + val.x*q.val.w + val.y*q.val.z - val.z*q.val.y;
		val.y = val.w*q.val.y + val.y*q.val.w + val.z*q.val.x - val.x*q.val.z;
		val.z = val.w*q.val.z + val.z*q.val.w + val.x*q.val.y - val.y*q.val.x;
		return *this;
	}    
	CUDA_FUNC_IN void buildFromAxisAngle(const float3& axis, float angle)
	{
		float radians = (angle/180.0f)*3.14159f;

		// cache this, since it is used multiple times below
		float sinThetaDiv2 = (float)sin( (radians/2.0f) );

		// now calculate the components of the quaternion	
		val.x = axis.x * sinThetaDiv2;
		val.y = axis.y * sinThetaDiv2;
		val.z = axis.z * sinThetaDiv2;

		val.w = (float)cos( (radians/2.0f) );
	}
	CUDA_FUNC_IN Quaternion conjugate() const { return Quaternion(-val.x, -val.y, -val.z, val.w); }
	CUDA_FUNC_IN float length() const
	{
		return ::length(val);
	}
	CUDA_FUNC_IN void normalize()
	{
		val = ::normalize(val);
	}
	CUDA_FUNC_IN Quaternion pow(float t)
	{
		Quaternion result(0,0,0,0);

		if ( fabs(val.w) < 0.9999 )
		{
			float alpha = (float)acos(val.w);
			float newAlpha = alpha * t;

			result.val.w = (float)cos( newAlpha);
			float fact = float( sin(newAlpha) / sin(alpha) );
			result.val.x *= fact;
			result.val.y *= fact;
			result.val.z *= fact;
		}
		return result;
	}
	CUDA_FUNC_IN float4x4 toMatrix()
	{
		float xx = val.x * val.x;
        float yy = val.y * val.y;
        float zz = val.z * val.z;
        float xy = val.x * val.y;
        float zw = val.z * val.w;
        float zx = val.z * val.x;
        float yw = val.y * val.w;
        float yz = val.y * val.z;
        float xw = val.x * val.w;
		return float4x4(
			1.0f - (2.0f * (yy + zz)), 2.0f * (xy + zw), 2.0f * (zx - yw), 0,
			2.0f * (xy - zw), 1.0f - (2.0f * (zz + xx)), 2.0f * (yz + xw), 0,
			2.0f * (zx + yw), 2.0f * (yz - xw), 1.0f - (2.0f * (yy + xx)), 0,
			0, 0, 0, 1
			);
	} 
};

CUDA_FUNC_IN Quaternion slerp(const Quaternion &q1, const Quaternion &q2, float t)
{
	Quaternion result, _q2 = q2;

	float cosOmega = q1.val.w * q2.val.w + q1.val.x * q2.val.x + q1.val.y * q2.val.y + q1.val.z * q2.val.z;

	if ( cosOmega < 0.0f )
	{
		_q2.val.x = -_q2.val.x;
		_q2.val.y = -_q2.val.y;
		_q2.val.z = -_q2.val.z;
		_q2.val.w = -_q2.val.w;
		cosOmega = -cosOmega;
	}

	float k0, k1;
	if ( cosOmega > 0.99999f )
	{
		k0 = 1.0f - t;
		k1 = t;
	}
	else
	{
		float sinOmega = (float)sqrt( 1.0f - cosOmega*cosOmega );
		float omega = (float)atan2( sinOmega, cosOmega );

		float invSinOmega = 1.0f/sinOmega;

		k0 = float( sin(((1.0f - t)*omega)) )*invSinOmega;
		k1 = float( sin(t*omega) )*invSinOmega;
	}
	result.val.x = q1.val.x * k0 + _q2.val.x * k1;
	result.val.y = q1.val.y * k0 + _q2.val.y * k1;
	result.val.z = q1.val.z * k0 + _q2.val.z * k1;
	result.val.w = q1.val.w * k0 + _q2.val.w * k1;

	return result;
}

__device__ inline unsigned int toABGR(const float4& v)
{
    return
        (unsigned int)(fminf(fmaxf(v.x, 0.0f), 1.0f) * 255.0f) |
        ((unsigned int)(fminf(fmaxf(v.y, 0.0f), 1.0f) * 255.0f) << 8) |
        ((unsigned int)(fminf(fmaxf(v.z, 0.0f), 1.0f) * 255.0f) << 16) |
        ((unsigned int)(fminf(fmaxf(v.w, 0.0f), 1.0f) * 255.0f) << 24);
}

__device__ inline unsigned int toABGR(const float3& v)
{
    return
        (unsigned int)(fminf(fmaxf(v.x, 0.0f), 1.0f) * 255.0f) |
        ((unsigned int)(fminf(fmaxf(v.y, 0.0f), 1.0f) * 255.0f) << 8) |
        ((unsigned int)(fminf(fmaxf(v.z, 0.0f), 1.0f) * 255.0f) << 16) |
        (255 << 24);
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

typedef uchar4 RGBCOL; 
#define toInt(x) (int((float)pow(clamp01(x),1.0f/1.2f)*255.0f+0.5f))
//#define toInt(x) (unsigned char(x * 255.0f))

CUDA_FUNC_IN RGBCOL Float4ToCOLORREF(const float4& c)
{
	return make_uchar4(toInt(c.x), toInt(c.y), toInt(c.z), toInt(c.w));
}

CUDA_FUNC_IN float4 COLORREFToFloat4(RGBCOL c)
{
	return make_float4((float)c.x / 255.0f, (float)c.y / 255.0f, (float)c.z / 255.0f, (float)c.w / 255.0f);
}

CUDA_FUNC_IN RGBCOL Float3ToCOLORREF(const float3& c)
{
	return make_uchar4(toInt(c.x), toInt(c.y), toInt(c.z), 255);
}

CUDA_FUNC_IN float3 COLORREFToFloat3(RGBCOL c)
{
	return make_float3((float)c.x / 255.0f, (float)c.y / 255.0f, (float)c.z / 255.0f);
}
#undef toInt

CUDA_FUNC_IN float frac(float f)
{
	return f - floorf(f);
}

#include "Int24.h"
#include "half.h"
typedef s10e5 half;

struct half2
{
	half x, y;
	CUDA_FUNC_IN half2() {}
	CUDA_FUNC_IN half2(float2& v)
	{
		x = half(v.x);
		y = half(v.y);
	}
	CUDA_FUNC_IN half2(float _x, float _y)
	{
		x = half(_x);
		y = half(_y);
	}
	CUDA_FUNC_IN float2 ToFloat2()
	{
		return make_float2(x.ToFloat(), y.ToFloat());
	}
};

struct half3
{
	half x, y, z;
	half3() {}
	CUDA_FUNC_IN half3(float3& v)
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
	CUDA_FUNC_IN float3 ToFloat3()
	{
		return make_float3(x.ToFloat(), y.ToFloat(), z.ToFloat());
	}
};

struct half4
{
	half x, y, z, w;
	half4() {}
	CUDA_FUNC_IN half4(float4& v)
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
	CUDA_FUNC_IN float4 ToFloat4()
	{
		return make_float4(x.ToFloat(), y.ToFloat(), z.ToFloat(), w.ToFloat());
	}
};

#include <limits>
template<typename T, typename V, int D, typename C, typename UC> struct NormalizedNBitFloatD
{
private:
	C data[D];
public:
	NormalizedNBitFloatD(){}
	NormalizedNBitFloatD(T args[D])
	{
		for(int i = 0; i < D; i++)
			data[i] = (C)((args[i] + 1.0f) * ((float)std::numeric_limits<C>::max()  / 2.0f));
	}
	NormalizedNBitFloatD(V& arg)
	{
		for(int i = 0; i < D; i++)
		{
			T* q = (T*)&arg + i;//we have trust in the compiler
			data[i] = (C)((*q + 1.0f) * ((float)std::numeric_limits<C>::max()  / 2.0f));
		}
	}
	CUDA_FUNC_IN V ToNative()
	{
		V r;
#define SET(sub, ind) r.sub = float(data[ind] - std::numeric_limits<UC>::max()) / float(std::numeric_limits<UC>::max());
		//r.sub = float(data[ind]) / float(std::numeric_limits<C>::max()) * 2.0f - 1.0f;
		SET(x, 0)
		if(D >= 2)
			SET(y, 1)
		if(D >= 3)
			SET(z, 2)
		//if(D >= 4)
		//	SET(w, 3)
		return r;
	}
};

typedef NormalizedNBitFloatD<float, float3, 3, unsigned char, char> Normalized24BitFloat3;
typedef NormalizedNBitFloatD<float, float3, 3, unsigned short, short> Normalized48BitFloat3;

struct Onb
{
	CUDA_FUNC_IN Onb()
	{

	}
	CUDA_FUNC_IN Onb( const float3& a)
	{
		m_normal = a;
		if (abs(a.x) > abs(a.y))
		{
			float invLen = 1.0f / sqrtf(a.x * a.x + a.z * a.z);
			m_tangent = make_float3(a.z * invLen, 0.0f, -a.x * invLen);
		}
		else
		{
			float invLen = 1.0f / sqrtf(a.y * a.y + a.z * a.z);
			m_tangent = make_float3(0.0f, a.z * invLen, -a.y * invLen);
		}
		m_binormal = cross(m_tangent, a);
	}
	CUDA_FUNC_IN float3 localToworld( const float3& p ) const
	{
		return p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
	}
	CUDA_FUNC_IN float3 worldTolocal(const float3& p) const
	{
		return make_float3(dot(m_tangent, p), dot(m_binormal, p), dot(m_normal, p));
	}
	CUDA_FUNC_IN Onb operator *(const float4x4& m) const
	{
		Onb r;
		r.m_normal = normalize(m.TransformNormal(m_normal));
		r.m_tangent = normalize(m.TransformNormal(m_tangent));
		r.m_binormal = normalize(m.TransformNormal(m_binormal));
		return r;
	}
	CUDA_FUNC_IN void RecalculateFromNormal(const float3& nor)
	{
		m_normal = nor;
		m_tangent = normalize(cross(nor, m_binormal));
		m_binormal = normalize(cross(nor, m_tangent));
	}

	float3 m_tangent;
	float3 m_binormal;
	float3 m_normal;
};

CUDA_FUNC_IN float3 SampleCosineHemisphere(const float3& n, float u, float v)
{
	float  theta = acos(sqrt(1.0f-u));
	float  phi = 2.0f * PI * v;

	float xs = sin(theta) * cos(phi);
	float ys = cos(theta);
	float zs = sin(theta) * sin(phi);

	float3 y = make_float3(n.x, n.y, n.z);
	float3 h = y;
	if (abs(h.x)<=abs(h.y) && abs(h.x)<=abs(h.z))
		h.x = 1.0;
	else if (abs(h.y)<=abs(h.x) && abs(h.y)<=abs(h.z))
		h.y = 1.0;
	else h.y = 1.0;

	float3 x = normalize(cross(h, y));
	float3 z = normalize(cross(x, y));

	float3 direction = xs * x + ys * y + zs * z;
	return normalize(direction);
}

CUDA_FUNC_IN float3 SampleUniformHemisphere(const float3& n, float u, float v, float w)
{
	float3 d = make_float3(u, v, w) * 2.0f - make_float3(1);
	return normalize(d * (dot(d, n) > 0 ? 1.0f : -1.0f));
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

#ifndef __CUDACC__
CUDA_FUNC_IN void sincos(float f, float* a, float* b)
{
	*a = sin(f);
	*b = cos(f);
}
#endif

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

CUDA_FUNC_IN float y(float3& v)
{
	const float YWeight[3] = { 0.212671f, 0.715160f, 0.072169f };
	return YWeight[0] * v.x + YWeight[1] * v.y + YWeight[2] * v.z;
}

template<typename T> CUDA_FUNC_IN T bilerp(const float2& uv, const T& lt, const T& rt, const T& ld, const T& rd)
{
	T a = lt + (rt - lt) * uv.x, b = ld + (rd - ld) * uv.x;
	return a + (b - a) * uv.y;
}

#include "Montecarlo.h"
#include "Ray.h"