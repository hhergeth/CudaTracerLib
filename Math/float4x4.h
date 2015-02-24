#pragma once

#include "Vector.h"

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
		r.row(0, Vec4f(xx, yx, zx, wx));
		r.row(1, Vec4f(xy, yy, zy, wy));
		r.row(2, Vec4f(xz, yz, zz, wz));
		r.row(3, Vec4f(xw, yw, zw, ww));
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
	CUDA_FUNC_IN Vec4f row(int i) const
	{
		return Vec4f(operator()(i, 0), operator()(i, 1), operator()(i, 2), operator()(i, 3));
	}
	CUDA_FUNC_IN void row(int i, const Vec4f& r)
	{
		operator()(i, 0) = r.x;
		operator()(i, 1) = r.y;
		operator()(i, 2) = r.z;
		operator()(i, 3) = r.w;
	}
	CUDA_FUNC_IN Vec4f col(int i) const
	{
		return Vec4f(operator()(0, i), operator()(1, i), operator()(2, i), operator()(3, i));
	}
	CUDA_FUNC_IN void col(int i, const Vec4f& r)
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

	CUDA_FUNC_IN Vec4f operator * (const Vec4f& a) const
	{
		return Vec4f(
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

	CUDA_FUNC_IN Vec3f TransformPoint(const Vec3f& p) const
	{
		Vec4f f = *this * Vec4f(p, 1.0f);
		return f.getXYZ() / f.w;
	}

	CUDA_FUNC_IN Vec3f TransformDirection(const Vec3f& d) const
	{
		Vec4f f = *this * Vec4f(d, 0.0f);
		return f.getXYZ();
	}

	CUDA_FUNC_IN Vec4f TransformTranspose(const Vec4f& a) const
	{
		return Vec4f(
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

	CUDA_FUNC_IN static float4x4 OuterProduct(const Vec4f& v)
	{
		float d[] = {v.x, v.y, v.z, v.w};
		float4x4 r;
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				r(i, j) = d[i] * d[j];
		return r;
	}

	CUDA_FUNC_IN static float4x4 RotationAxis(const Vec3f& _axis, const float angle)
	{
		Vec3f naxis = normalize(_axis);
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

	CUDA_FUNC_IN static float4x4 Translate(const Vec3f& t)
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

	CUDA_FUNC_IN static float4x4 Scale(const Vec3f& s)
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
		return Scale(Vec3f(1.0f, 1.0f, 1.0f / (clipFar - clipNear))) %
			Translate(Vec3f(0.0f, 0.0f, -clipNear));
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

	CUDA_FUNC_IN static float4x4 lookAt(const Vec3f &p, const Vec3f &t, const Vec3f &up)
	{
		Vec3f dir = normalize(t - p);
		Vec3f left = normalize(cross(up, dir));
		Vec3f newUp = cross(dir, left);
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
		mat.col(0, Vec4f(2.0f / w, 0, 0, 0));
		mat.col(1, Vec4f(0, 2.0f / h, 0, 0));
		mat.col(2, Vec4f(0, 0, 1.0f / (f - n), 0));
		mat.col(3, Vec4f(0, 0, n / (n - f), 1));
		return mat;
	}

	CUDA_FUNC_IN static float4x4 Perspective(float fov, float asp, float n, float f)
	{
		float cosfov = cosf(0.5f * fov), sinfov = sinf(0.5f * fov), h = cosfov / sinfov, w = h / asp;
		float4x4 mat = float4x4::Identity();
		mat.col(0, Vec4f(w, 0, 0, 0));
		mat.col(1, Vec4f(0, h, 0, 0));
		mat.col(2, Vec4f(0, 0, -(f + n) / (f - n), 1));
		mat.col(3, Vec4f(0, 0, -(n * f) / (f - n), 0));
		return mat;
	}

	CUDA_FUNC_IN Vec3f Translation() const
	{
		return TransformPoint(Vec3f(0.0f));
	}
	CUDA_FUNC_IN Vec3f Scale() const
	{
		return Vec3f(length(col(0).getXYZ()), length(col(1).getXYZ()), length(col(2).getXYZ()));
	}
	CUDA_FUNC_IN Vec3f Forward() const
	{
		return TransformDirection(Vec3f(0, 0, 1));
	}
	CUDA_FUNC_IN Vec3f Right() const
	{
		return TransformDirection(Vec3f(1, 0, 0));
	}
	CUDA_FUNC_IN Vec3f Up() const
	{
		return TransformDirection(Vec3f(0, 1, 0));
	}

	friend std::ostream& operator<< (std::ostream & os, const float4x4& rhs);
};

inline std::ostream& operator<< (std::ostream & os, const float4x4& rhs)
{
	os << "[" << rhs.row(0)
		<< "\n " << rhs.row(1)
		<< "\n " << rhs.row(2)
		<< "\n " << rhs.row(3) << "\n]";
	return os;
}