#pragma once

#include "cutil_math.h"

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
		
		static float4x4 LookAt(const float3& _From, const float3& _To, const float3& _Up)
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

		static float4x4 Perspective(float fov, float asp, float n, float f)
		{
			float cosfov = cosf(0.5f * fov), sinfov = sinf(0.5f * fov), h = cosfov / sinfov, w = h / asp;
			float4x4 mat=float4x4::Identity();
			mat.X = make_float4(w,0,0,0);
			mat.Y = make_float4(0,h,0,0);
			mat.Z = make_float4(0,0,f / (f - n),1);
			mat.W = make_float4(0,0,- (n * f) / (f - n),0);
			return mat;
		}

		static float4x4 Orthographic(float w, float h, float n, float f)
		{
			float4x4 mat=float4x4::Identity();
			mat.X = make_float4(2.0f / w, 0, 0, 0);
			mat.Y = make_float4(0, 2.0f / h, 0, 0);
			mat.Z = make_float4(0, 0, 1.0f / (f - n), 0);
			mat.W = make_float4(0, 0, n / (n - f), 1);
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

		CUDA_FUNC_IN float3 Translation() const
		{
			return make_float3(W.x, W.y, W.z);
		}
		CUDA_FUNC_IN void Translation(float3 t)
		{
			W.x = t.x;
			W.y = t.y;
			W.z = t.z;
		}

		CUDA_FUNC_IN float3 Scale() const
		{
			return make_float3(length(!X), length(!Y), length(!Z));
		}

		CUDA_FUNC_IN float3 Forward() const
		{
			return make_float3(Z.x, Z.y, Z.z);
		}
		CUDA_FUNC_IN float3 Right() const
		{
			return make_float3(X.x, X.y, X.z);
		}
		CUDA_FUNC_IN float3 Up() const
		{
			return make_float3(Y.x, Y.y, Y.z);
		}
	};