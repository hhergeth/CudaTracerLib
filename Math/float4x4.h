#pragma once

#include "cutil_math.h"

class CUDA_ALIGN(16) float4x4
{
	float data[16];
	CUDA_FUNC_IN int idx(int i, int j) const
	{
		return i * 4 + j;
	}
public:
	CUDA_FUNC_IN float4x4()
	{
	}

	CUDA_FUNC_IN static float4x4 As(float xx, float yx, float zx, float wx,
									float xy, float yy, float zy, float wy,
									float xz, float yz, float zz, float wz,
									float xw, float yw, float zw, float ww)
	{
		float4x4 r;
		r.row(0, make_float4(xx, yx, zx, wx));
		r.row(1, make_float4(xy, yy, zy, wy));
		r.row(2, make_float4(xz, yz, zz, wz));
		r.row(3, make_float4(xw, yw, zw, ww));
		return r;
	}

	CUDA_FUNC_IN void zeros()
	{
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				operator()(i, j) = 0;
	}

	//access
	CUDA_FUNC_IN float operator()(int i, int j) const
	{
		return data[idx(i, j)];
	}
	CUDA_FUNC_IN float& operator()(int i, int j)
	{
		return data[idx(i, j)];
	}
	CUDA_FUNC_IN float4 row(int i) const
	{
		return make_float4(operator()(i, 0), operator()(i, 1), operator()(i, 2), operator()(i, 3));
	}
	CUDA_FUNC_IN void row(int i, const float4& r)
	{
		operator()(i, 0) = r.x;
		operator()(i, 1) = r.y;
		operator()(i, 2) = r.z;
		operator()(i, 3) = r.w;
	}
	CUDA_FUNC_IN float4 col(int i) const
	{
		return make_float4(operator()(0, i), operator()(1, i), operator()(2, i), operator()(3, i));
	}
	CUDA_FUNC_IN void col(int i, const float4& r)
	{
		operator()(0, i) = r.x;
		operator()(1, i) = r.y;
		operator()(2, i) = r.z;
		operator()(3, i) = r.w;
	}

	//operators
	/*CUDA_FUNC_IN void operator *= (const float4x4& b)
	{
		float4x4 a = *this;
		*this = a * b;
	}*/

	CUDA_FUNC_IN float4x4 operator + (const float4x4& a) const
	{
		float4x4 r;
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				r(i, j) = operator()(i, j) + a(i, j);
		return r;
	}
	
	CUDA_FUNC_IN float4x4 operator % (const float4x4& a) const
	{
		float4x4 r;
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				r(i, j) = dot(row(i), a.col(j));
		return r;
	}

	CUDA_FUNC_IN float4 operator * (const float4& a) const
	{
		return make_float4(
			dot(row(0), a),
			dot(row(1), a),
			dot(row(2), a),
			dot(row(3), a)
			);
	}

	CUDA_FUNC_IN float4x4 operator * (const float a) const
	{
		float4x4 r;
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				r(i, j) = operator()(i, j) + a;
		return r;
	}

	CUDA_FUNC_IN float3 TransformPoint(const float3& p) const
	{
		float4 f = *this * make_float4(p, 1.0f);
		return (!f) / f.w;
	}

	CUDA_FUNC_IN float3 TransformDirection(const float3& d) const
	{
		float4 f = *this * make_float4(d, 0.0f);
		return !f;
	}

	CUDA_FUNC_IN float4 TransformTranspose(const float4& a) const
	{
		return make_float4(
			dot(col(0), a),
			dot(col(1), a),
			dot(col(2), a),
			dot(col(3), a)
			);
	}

	CUDA_FUNC_IN float4x4 transpose() const
	{
		float4x4 r;
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				r(i, j) = operator()(j, i);
		return r;
	}

	CUDA_DEVICE CUDA_HOST float4x4 inverse() const;

	CUDA_FUNC_IN static float4x4 RotateX(float a)
	{
		float4x4 r = float4x4::Identity();
		float cosa = cosf(a), sina = sin(a);
		r(1, 1) = cosa;
		r(1, 2) = -sina;
		r(2, 1) = sina;
		r(2, 2) = cosa;
		return r;
	}

	CUDA_FUNC_IN static float4x4 RotateY(float a)
	{
		float4x4 r = float4x4::Identity();
		float cosa = cosf(a), sina = sin(a);
		r(0, 0) = cosa;
		r(0, 2) = sina;
		r(2, 0) = -sina;
		r(2, 2) = cosa;
		return r;
	}

	CUDA_FUNC_IN static float4x4 RotateZ(float a)
	{
		float4x4 r = float4x4::Identity();
		float cosa = cosf(a), sina = sin(a);
		r(0, 0) = cosa;
		r(0, 1) = -sina;
		r(1, 0) = sina;
		r(1, 1) = cosa;
		return r;
	}

	CUDA_FUNC_IN static float4x4 OuterProduct(const float4& v)
	{
		float d[] = {v.x, v.y, v.z, v.w};
		float4x4 r;
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				r(i, j) = d[i] * d[j];
		return r;
	}

	CUDA_FUNC_IN static float4x4 RotationAxis(const float3& _axis, const float angle)
	{
		float3 naxis = normalize(_axis);
		float sinTheta, cosTheta;
		sincos(angle, &sinTheta, &cosTheta);
		float4x4 result;
		result(0, 0) = naxis.x * naxis.x + (1.0f - naxis.x * naxis.x) * cosTheta;
		result(0, 1) = naxis.x * naxis.y * (1.0f - cosTheta) - naxis.z * sinTheta;
		result(0, 2) = naxis.x * naxis.z * (1.0f - cosTheta) + naxis.y * sinTheta;
		result(0, 3) = 0;

		result(1, 0) = naxis.x * naxis.y * (1.0f - cosTheta) + naxis.z * sinTheta;
		result(1, 1) = naxis.y * naxis.y + (1.0f - naxis.y * naxis.y) * cosTheta;
		result(1, 2) = naxis.y * naxis.z * (1.0f - cosTheta) - naxis.x * sinTheta;
		result(1, 3) = 0;

		result(2, 0) = naxis.x * naxis.z * (1.0f - cosTheta) - naxis.y * sinTheta;
		result(2, 1) = naxis.y * naxis.z * (1.0f - cosTheta) + naxis.x * sinTheta;
		result(2, 2) = naxis.z * naxis.z + (1.0f - naxis.z * naxis.z) * cosTheta;
		result(2, 3) = 0;

		result(3, 0) = 0;
		result(3, 1) = 0;
		result(3, 2) = 0;
		result(3, 3) = 1;

		return result;
	}

	CUDA_FUNC_IN static const float4x4 Identity()
	{
		float4x4 r;
		r.zeros();
		r(0, 0) = r(1, 1) = r(2, 2) = r(3, 3) = 1.0f;
		return r;
	}

	CUDA_FUNC_IN static float4x4 Translate(float3 t)
	{
		return Translate(t.x, t.y, t.z);
	}

	CUDA_FUNC_IN static float4x4 Translate(float x, float y, float z)
	{
		float4x4 r = float4x4::Identity();
		r(0, 3) = x;
		r(1, 3) = y;
		r(2, 3) = z;
		return r;
	}

	CUDA_FUNC_IN static float4x4 Scale(float3 s)
	{
		return Scale(s.x, s.y, s.z);
	}

	CUDA_FUNC_IN static float4x4 Scale(float x, float y, float z)
	{
		float4x4 r = float4x4::Identity();
		r(0, 0) = x;
		r(1, 1) = y;
		r(2, 2) = z;
		return r;
	}

	CUDA_FUNC_IN static float4x4 Perspective(float fov, float clipNear, float clipFar)
	{
		float recip = 1.0f / (clipFar - clipNear);

		/* Perform a scale so that the field of view is mapped
		* to the interval [-1, 1] */
		float cot = 1.0f / tanf(fov / 2.0f);

		float4x4 trafo = float4x4::As(
			cot, 0, 0, 0,
			0, cot, 0, 0,
			0, 0, clipFar * recip, -clipNear * clipFar * recip,
			0, 0, 1, 0
			);
		return trafo;
	}

	CUDA_FUNC_IN static float4x4 glPerspective(float fov, float clipNear, float clipFar)
	{
		float recip = 1.0f / (clipNear - clipFar);
		float cot = 1.0f / tanf(fov / 2.0f);

		float4x4 trafo = float4x4::As(
			cot, 0, 0, 0,
			0, cot, 0, 0,
			0, 0, (clipFar + clipNear) * recip, 2 * clipFar * clipNear * recip,
			0, 0, -1, 0
			);

		return trafo;
	}

	CUDA_FUNC_IN static float4x4 glFrustum(float left, float right, float bottom, float top, float nearVal, float farVal)
	{
		float invFMN = 1 / (farVal - nearVal);
		float invTMB = 1 / (top - bottom);
		float invRML = 1 / (right - left);

		float4x4 trafo = float4x4::As(
			2 * nearVal*invRML, 0, (right + left)*invRML, 0,
			0, 2 * nearVal*invTMB, (top + bottom)*invTMB, 0,
			0, 0, -(farVal + nearVal) * invFMN, -2 * farVal*nearVal*invFMN,
			0, 0, -1, 0
			);

		return trafo;
	}

	CUDA_FUNC_IN static float4x4 orthographic(float clipNear, float clipFar)
	{
		return Scale(make_float3(1.0f, 1.0f, 1.0f / (clipFar - clipNear))) %
			Translate(make_float3(0.0f, 0.0f, -clipNear));
	}

	CUDA_FUNC_IN static float4x4 glOrthographic(float clipNear, float clipFar)
	{
		float a = -2.0f / (clipFar - clipNear),
			b = -(clipFar + clipNear) / (clipFar - clipNear);

		float4x4 trafo = float4x4::As(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, a, b,
			0, 0, 0, 1
			);
		return trafo;
	}

	CUDA_FUNC_IN static float4x4 glOrthographic(float clipLeft, float clipRight,
		float clipBottom, float clipTop, float clipNear, float clipFar)
	{
		float fx = 2.0f / (clipRight - clipLeft),
			fy = 2.0f / (clipTop - clipBottom),
			fz = -2.0f / (clipFar - clipNear),
			tx = -(clipRight + clipLeft) / (clipRight - clipLeft),
			ty = -(clipTop + clipBottom) / (clipTop - clipBottom),
			tz = -(clipFar + clipNear) / (clipFar - clipNear);

		float4x4 trafo = float4x4::As(
			fx, 0, 0, tx,
			0, fy, 0, ty,
			0, 0, fz, tz,
			0, 0, 0, 1
			);

		return trafo;
	}

	CUDA_FUNC_IN static float4x4 lookAt(const float3 &p, const float3 &t, const float3 &up)
	{
		float3 dir = normalize(t - p);
		float3 left = normalize(cross(up, dir));
		float3 newUp = cross(dir, left);
		float4x4 result;
		result(0, 0) = left.x;  result(1, 0) = left.y;  result(2, 0) = left.z;  result(3, 0) = 0;
		result(0, 1) = newUp.x; result(1, 1) = newUp.y; result(2, 1) = newUp.z; result(3, 1) = 0;
		result(0, 2) = dir.x;   result(1, 2) = dir.y;   result(2, 2) = dir.z;   result(3, 2) = 0;
		result(0, 3) = p.x;     result(1, 3) = p.y;     result(2, 3) = p.z;     result(3, 3) = 1;
		return result;
	}

	CUDA_FUNC_IN static float4x4 Orthographic(float w, float h, float n, float f)
	{
		float4x4 mat = float4x4::Identity();
		mat.col(0, make_float4(2.0f / w, 0, 0, 0));
		mat.col(1, make_float4(0, 2.0f / h, 0, 0));
		mat.col(2, make_float4(0, 0, 1.0f / (f - n), 0));
		mat.col(3, make_float4(0, 0, n / (n - f), 1));
		return mat;
	}

	CUDA_FUNC_IN static float4x4 Perspective(float fov, float asp, float n, float f)
	{
		float cosfov = cosf(0.5f * fov), sinfov = sinf(0.5f * fov), h = cosfov / sinfov, w = h / asp;
		float4x4 mat = float4x4::Identity();
		mat.col(0, make_float4(w, 0, 0, 0));
		mat.col(1, make_float4(0, h, 0, 0));
		mat.col(2, make_float4(0, 0, -(f + n) / (f - n), 1));
		mat.col(3, make_float4(0, 0, -(n * f) / (f - n), 0));
		return mat;
	}

	CUDA_FUNC_IN float3 Translation() const
	{
		return TransformPoint(make_float3(0.0f));
	}
	CUDA_FUNC_IN float3 Scale() const
	{
		return make_float3(length(!col(0)), length(!col(1)), length(!col(2)));
	}
	CUDA_FUNC_IN float3 Forward() const
	{
		return TransformDirection(make_float3(0, 0, 1));
	}
	CUDA_FUNC_IN float3 Right() const
	{
		return TransformDirection(make_float3(1, 0, 0));
	}
	CUDA_FUNC_IN float3 Up() const
	{
		return TransformDirection(make_float3(0, 1, 0));
	}
};
/*
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
		
		CUDA_FUNC_IN static float4x4 LookAt(const float3& _From, const float3& _To, const float3& _Up)
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

		CUDA_FUNC_IN static float4x4 Perspective(float fov, float asp, float n, float f)
		{
			float cosfov = cosf(0.5f * fov), sinfov = sinf(0.5f * fov), h = cosfov / sinfov, w = h / asp;
			float4x4 mat=float4x4::Identity();
			mat.X = make_float4(w,0,0,0);
			mat.Y = make_float4(0,h,0,0);
			mat.Z = make_float4(0,0,f / (f - n),1);
			mat.W = make_float4(0,0,- (n * f) / (f - n),0);
			return mat;
		}

		CUDA_FUNC_IN static float4x4 Orthographic(float w, float h, float n, float f)
		{
			float4x4 mat=float4x4::Identity();
			mat.X = make_float4(2.0f / w, 0, 0, 0);
			mat.Y = make_float4(0, 2.0f / h, 0, 0);
			mat.Z = make_float4(0, 0, 1.0f / (f - n), 0);
			mat.W = make_float4(0, 0, n / (n - f), 1);
			return mat;
		}

		CUDA_FUNC_IN static float4x4 Lerp(const float4x4&a, const float4x4&b, float t)
		{
			float t2 = 1.0f - t;
			float4x4 r;
			for(int i = 0; i < 4; i++)
				r.Row[i] = a.Row[i] * t2 + b.Row[i] * t;
			return r;
		}

		CUDA_FUNC_IN static float4x4 RotateX( float a )
		{
			float x = cosf( a );
			float y = sinf( a );
			return float4x4(1, 0, 0, 0,	0, x, y, 0,	0, -y, x, 0, 0, 0, 0, 1 );
		}
		CUDA_FUNC_IN static float4x4 RotateY( float a )
		{
			float x = cosf( a );
			float y = sinf( a );
			return float4x4( x, 0, -y, 0, 0, 1, 0, 0, y, 0, x, 0, 0, 0, 0, 1 );
		}
		CUDA_FUNC_IN static float4x4 RotateZ( float a )
		{
			float x = cosf( a );
			float y = sinf( a );
			return float4x4( x, y, 0, 0, -y, x, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 );
		}
		CUDA_FUNC_IN static float4x4 Rotate( float3 _XYZ )
		{
			return RotateX(_XYZ.x)*RotateY(_XYZ.y)*RotateZ(_XYZ.z);
		}
		CUDA_FUNC_IN static float4x4 RotationAxis(const float3& _axis, const float angle)
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

		CUDA_FUNC_IN static const float4x4 Identity()
		{
			return float4x4( 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
		}
		
		CUDA_FUNC_IN static float4x4 Translate( float3 t )
		{
			return float4x4( 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, t.x, t.y, t.z, 1 );
		}
		
		CUDA_FUNC_IN static float4x4 Translate( float x, float y, float z )
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

		CUDA_FUNC_IN static float4x4 Scale( float3 s )
		{
			return float4x4( s.x, 0, 0, 0, 0, s.y, 0, 0, 0, 0, s.z, 0, 0, 0, 0, 1 );
		}
		CUDA_FUNC_IN static float4x4 Scale( float s )
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
	};*/