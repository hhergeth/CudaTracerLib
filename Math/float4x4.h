#pragma once

#include "Vector.h"

//Matrix class for vectors in R^3x1 i.e. column vectors.

namespace CudaTracerLib {

struct OrthogonalAffineMap;
template<> struct NormalizedT<OrthogonalAffineMap>;

struct CUDA_ALIGN(16) float4x4
{
	float data[16];
	CUDA_FUNC_IN static int idx(int i, int j)
	{
		return i * 4 + j;
	}
public:
	CUDA_FUNC_IN float4x4()
	{
	}

	CUDA_FUNC_IN static float4x4 As(
		float xx, float yx, float zx, float wx,
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

	CUDA_FUNC_IN static float4x4 As(const Vec4f& col0, const Vec4f& col1, const Vec4f& col2, const Vec4f& col3)
	{
		float4x4 r;
		r.col(0, col0);	r.col(1, col1);	r.col(2, col2);	r.col(3, col3);
		return r;
	}

	CUDA_FUNC_IN static float4x4 As(const Vec3f& col0, const Vec3f& col1, const Vec3f& col2, const Vec3f& col3)
	{
		float4x4 r;
		r.col(0, Vec4f(col0, 0.0f));	r.col(1, Vec4f(col1, 0.0f));	r.col(2, Vec4f(col2, 0.0f));	r.col(3, Vec4f(col3, 1.0f));
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

	//algebra
	CUDA_FUNC_IN Vec3f Translation() const
	{
		return TransformPoint(Vec3f(0.0f));
	}
	CUDA_FUNC_IN Vec3f Scale() const
	{
		return Vec3f(col(0).getXYZ().length(), col(1).getXYZ().length(), col(2).getXYZ().length());
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

	//returns the value of the frobenius norm
	CUDA_FUNC_IN float length() const
	{
		return math::sqrt(row(0).lenSqr() + row(1).lenSqr() + row(2).lenSqr() + row(3).lenSqr());
	}

	CUDA_FUNC_IN Vec3f TransformPoint(const Vec3f& p) const;

	CUDA_FUNC_IN Vec3f TransformDirection(const Vec3f& d) const;

	CUDA_FUNC_IN float4x4 transpose() const
	{
		float4x4 r;
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				r(i, j) = operator()(j, i);
		return r;
	}

	CUDA_FUNC_IN float4x4 inverse() const
	{
		float4x4 Q = *this;
		float m00 = Q(0, 0), m01 = Q(0, 1), m02 = Q(0, 2), m03 = Q(0, 3);
		float m10 = Q(1, 0), m11 = Q(1, 1), m12 = Q(1, 2), m13 = Q(1, 3);
		float m20 = Q(2, 0), m21 = Q(2, 1), m22 = Q(2, 2), m23 = Q(2, 3);
		float m30 = Q(3, 0), m31 = Q(3, 1), m32 = Q(3, 2), m33 = Q(3, 3);

		float v0 = m20 * m31 - m21 * m30;
		float v1 = m20 * m32 - m22 * m30;
		float v2 = m20 * m33 - m23 * m30;
		float v3 = m21 * m32 - m22 * m31;
		float v4 = m21 * m33 - m23 * m31;
		float v5 = m22 * m33 - m23 * m32;

		float t00 = +(v5 * m11 - v4 * m12 + v3 * m13);
		float t10 = -(v5 * m10 - v2 * m12 + v1 * m13);
		float t20 = +(v4 * m10 - v2 * m11 + v0 * m13);
		float t30 = -(v3 * m10 - v1 * m11 + v0 * m12);

		float invDet = 1 / (t00 * m00 + t10 * m01 + t20 * m02 + t30 * m03);

		float d00 = t00 * invDet;
		float d10 = t10 * invDet;
		float d20 = t20 * invDet;
		float d30 = t30 * invDet;

		float d01 = -(v5 * m01 - v4 * m02 + v3 * m03) * invDet;
		float d11 = +(v5 * m00 - v2 * m02 + v1 * m03) * invDet;
		float d21 = -(v4 * m00 - v2 * m01 + v0 * m03) * invDet;
		float d31 = +(v3 * m00 - v1 * m01 + v0 * m02) * invDet;

		v0 = m10 * m31 - m11 * m30;
		v1 = m10 * m32 - m12 * m30;
		v2 = m10 * m33 - m13 * m30;
		v3 = m11 * m32 - m12 * m31;
		v4 = m11 * m33 - m13 * m31;
		v5 = m12 * m33 - m13 * m32;

		float d02 = +(v5 * m01 - v4 * m02 + v3 * m03) * invDet;
		float d12 = -(v5 * m00 - v2 * m02 + v1 * m03) * invDet;
		float d22 = +(v4 * m00 - v2 * m01 + v0 * m03) * invDet;
		float d32 = -(v3 * m00 - v1 * m01 + v0 * m02) * invDet;

		v0 = m21 * m10 - m20 * m11;
		v1 = m22 * m10 - m20 * m12;
		v2 = m23 * m10 - m20 * m13;
		v3 = m22 * m11 - m21 * m12;
		v4 = m23 * m11 - m21 * m13;
		v5 = m23 * m12 - m22 * m13;

		float d03 = -(v5 * m01 - v4 * m02 + v3 * m03) * invDet;
		float d13 = +(v5 * m00 - v2 * m02 + v1 * m03) * invDet;
		float d23 = -(v4 * m00 - v2 * m01 + v0 * m03) * invDet;
		float d33 = +(v3 * m00 - v1 * m01 + v0 * m02) * invDet;

		return float4x4::As(
			d00, d01, d02, d03,
			d10, d11, d12, d13,
			d20, d21, d22, d23,
			d30, d31, d32, d33);
	}

	//geometric constructor functions
	CUDA_FUNC_IN static NormalizedT<OrthogonalAffineMap> RotateX(float a);

	CUDA_FUNC_IN static NormalizedT<OrthogonalAffineMap> RotateY(float a);

	CUDA_FUNC_IN static NormalizedT<OrthogonalAffineMap> RotateZ(float a);

	CUDA_FUNC_IN static NormalizedT<OrthogonalAffineMap> RotationAxis(const NormalizedT<Vec3f>& _axis, const float angle);

	CUDA_FUNC_IN static float4x4 OuterProduct(const Vec4f& a, const Vec4f& b)
	{
		float4x4 r;
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				r(i, j) = a[i] * b[j];
		return r;
	}

	CUDA_FUNC_IN static float4x4 Identity()
	{
		float4x4 r;
		r.zeros();
		r(0, 0) = r(1, 1) = r(2, 2) = r(3, 3) = 1.0f;
		return r;
	}

	CUDA_FUNC_IN static NormalizedT<OrthogonalAffineMap> Translate(const Vec3f& t);

	CUDA_FUNC_IN static NormalizedT<OrthogonalAffineMap> Translate(float x, float y, float z);

	CUDA_FUNC_IN static OrthogonalAffineMap Scale(const Vec3f& s);

	CUDA_FUNC_IN static OrthogonalAffineMap Scale(float x, float y, float z);

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

	CUDA_FUNC_IN static float4x4 orthographic(float clipNear, float clipFar);

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

	CUDA_FUNC_IN static NormalizedT<OrthogonalAffineMap> lookAt(const Vec3f &p, const Vec3f &t, const Vec3f &up);

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

	friend std::ostream& operator<< (std::ostream & os, const float4x4& rhs)
	{
		os << "[" << rhs.row(0)
			<< "\n " << rhs.row(1)
			<< "\n " << rhs.row(2)
			<< "\n " << rhs.row(3) << "\n]";
		return os;
	}
};

//operators
CUDA_FUNC_IN float4x4 operator + (const float4x4& lhs, const float4x4& rhs)
{
	float4x4 r;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			r(i, j) = lhs(i, j) + rhs(i, j);
	return r;
}

CUDA_FUNC_IN float4x4 operator - (const float4x4& lhs, const float4x4& rhs)
{
	float4x4 r;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			r(i, j) = lhs(i, j) - rhs(i, j);
	return r;
}

CUDA_FUNC_IN float4x4 operator % (const float4x4& lhs, const float4x4& rhs)
{
	float4x4 r;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			r(i, j) = dot(lhs.row(i), rhs.col(j));
	return r;
}

CUDA_FUNC_IN Vec4f operator * (const float4x4& lhs, const Vec4f& rhs)
{
	return Vec4f(
		dot(lhs.row(0), rhs),
		dot(lhs.row(1), rhs),
		dot(lhs.row(2), rhs),
		dot(lhs.row(3), rhs)
		);
}

CUDA_FUNC_IN float4x4 operator * (float lhs, const float4x4& rhs)
{
	float4x4 r;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			r(i, j) = rhs(i, j) * lhs;
	return r;
}

CUDA_FUNC_IN float4x4 operator * (const float4x4& lhs, float rhs)
{
	return rhs * lhs;
}

Vec3f float4x4::TransformPoint(const Vec3f& p) const
{
	Vec4f f = *this * Vec4f(p, 1.0f);
	return f.getXYZ() / f.w;
}

Vec3f float4x4::TransformDirection(const Vec3f& d) const
{
	Vec4f f = *this * Vec4f(d, 0.0f);
	return f.getXYZ();
}

//The 3x3 rotational part is assumed to be orthogonal, the homogeneous part must equal zero
struct OrthogonalAffineMap : public float4x4
{
	CUDA_FUNC_IN OrthogonalAffineMap()
	{

	}

	CUDA_FUNC_IN explicit OrthogonalAffineMap(const float4x4& m)
		: float4x4(m)
	{

	}

	CUDA_FUNC_IN Vec3f TransformDirectionTranspose(const Vec3f& d) const
	{
		return Vec3f(dot(d, col(0).getXYZ()), dot(d, col(1).getXYZ()), dot(d, col(2).getXYZ()));
	}

	CUDA_FUNC_IN Vec3f TransformPointTranspose(const Vec3f& d) const
	{
		return TransformDirectionTranspose(d) - TransformDirectionTranspose(Translation());
	}

	CUDA_FUNC_IN OrthogonalAffineMap inverse() const
	{
		return transpose();
	}

	CUDA_FUNC_IN OrthogonalAffineMap transpose() const
	{
		return OrthogonalAffineMap(float4x4::As(row(0).getXYZ(), row(1).getXYZ(), row(2).getXYZ(), -TransformDirectionTranspose(Translation())));
	}

	CUDA_FUNC_IN static OrthogonalAffineMap Identity()
	{
		return OrthogonalAffineMap(float4x4::Identity());
	}
};

CUDA_FUNC_IN OrthogonalAffineMap operator % (const OrthogonalAffineMap& lhs, const OrthogonalAffineMap& rhs)
{
	return OrthogonalAffineMap(*(float4x4*)&lhs % *(float4x4*)&rhs);
}

template<> struct NormalizedT<OrthogonalAffineMap> : public OrthogonalAffineMap
{
	CUDA_FUNC_IN NormalizedT()
	{

	}

	CUDA_FUNC_IN explicit NormalizedT(const OrthogonalAffineMap& v)
		: OrthogonalAffineMap(v)
	{

	}

	CUDA_FUNC_IN NormalizedT<OrthogonalAffineMap> inverse() const
	{
		return transpose();
	}

	CUDA_FUNC_IN NormalizedT<OrthogonalAffineMap> transpose() const
	{
		return NormalizedT<OrthogonalAffineMap>(OrthogonalAffineMap::transpose());
	}

	CUDA_FUNC_IN NormalizedT<Vec3f> TransformDirection(const NormalizedT<Vec3f>& d) const
	{
		return NormalizedT<Vec3f>(TransformDirection((Vec3f)d));
	}

	CUDA_FUNC_IN NormalizedT<Vec3f> TransformDirectionTranspose(const NormalizedT<Vec3f>& d) const
	{
		return NormalizedT<Vec3f>(TransformDirectionTranspose((Vec3f)d));
	}

	//these functions are hidden by the previous ones
	CUDA_FUNC_IN Vec3f TransformDirection(const Vec3f& d) const
	{
		return OrthogonalAffineMap::TransformDirection(d);
	}

	CUDA_FUNC_IN Vec3f TransformDirectionTranspose(const Vec3f& d) const
	{
		return OrthogonalAffineMap::TransformDirectionTranspose(d);
	}

	CUDA_FUNC_IN static NormalizedT<OrthogonalAffineMap> Identity()
	{
		return NormalizedT<OrthogonalAffineMap>(OrthogonalAffineMap::Identity());
	}

	CUDA_FUNC_IN NormalizedT<Vec3f> Right() const
	{
		return TransformDirection(NormalizedT<Vec3f>(1.0f, 0.0f, 0.0f));
	}
	CUDA_FUNC_IN NormalizedT<Vec3f> Up() const
	{
		return TransformDirection(NormalizedT<Vec3f>(0.0f, 1.0f, 0.0f));
	}
	CUDA_FUNC_IN NormalizedT<Vec3f> Forward() const
	{
		return TransformDirection(NormalizedT<Vec3f>(0.0f, 0.0f, 1.0f));
	}
};

CUDA_FUNC_IN NormalizedT<OrthogonalAffineMap> operator % (const NormalizedT<OrthogonalAffineMap>& lhs, const NormalizedT<OrthogonalAffineMap>& rhs)
{
	return NormalizedT<OrthogonalAffineMap>(*(OrthogonalAffineMap*)&lhs % *(OrthogonalAffineMap*)&rhs);
}

NormalizedT<OrthogonalAffineMap> float4x4::RotateX(float a)
{
	auto r = NormalizedT<OrthogonalAffineMap>::Identity();
	float cosa = cosf(a), sina = sin(a);
	r(1, 1) = cosa;
	r(1, 2) = -sina;
	r(2, 1) = sina;
	r(2, 2) = cosa;
	return r;
}

NormalizedT<OrthogonalAffineMap> float4x4::RotateY(float a)
{
	auto r = NormalizedT<OrthogonalAffineMap>::Identity();
	float cosa = cosf(a), sina = sin(a);
	r(0, 0) = cosa;
	r(0, 2) = sina;
	r(2, 0) = -sina;
	r(2, 2) = cosa;
	return r;
}

NormalizedT<OrthogonalAffineMap> float4x4::RotateZ(float a)
{
	auto r = NormalizedT<OrthogonalAffineMap>::Identity();
	float cosa = cosf(a), sina = sin(a);
	r(0, 0) = cosa;
	r(0, 1) = -sina;
	r(1, 0) = sina;
	r(1, 1) = cosa;
	return r;
}

NormalizedT<OrthogonalAffineMap> float4x4::RotationAxis(const NormalizedT<Vec3f>& naxis, const float angle)
{
	float sinTheta, cosTheta;
	sincos(angle, &sinTheta, &cosTheta);
	NormalizedT<OrthogonalAffineMap> result;
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

NormalizedT<OrthogonalAffineMap> float4x4::Translate(const Vec3f& t)
{
	return Translate(t.x, t.y, t.z);
}

NormalizedT<OrthogonalAffineMap> float4x4::Translate(float x, float y, float z)
{
	auto r = NormalizedT<OrthogonalAffineMap>::Identity();
	r(0, 3) = x;
	r(1, 3) = y;
	r(2, 3) = z;
	return r;
}

OrthogonalAffineMap float4x4::Scale(const Vec3f& s)
{
	return Scale(s.x, s.y, s.z);
}

OrthogonalAffineMap float4x4::Scale(float x, float y, float z)
{
	auto r = OrthogonalAffineMap::Identity();
	r(0, 0) = x;
	r(1, 1) = y;
	r(2, 2) = z;
	return r;
}

NormalizedT<OrthogonalAffineMap> float4x4::lookAt(const Vec3f &p, const Vec3f &t, const Vec3f &up)
{
	Vec3f dir = normalize(t - p);
	Vec3f left = normalize(cross(up, dir));
	Vec3f newUp = cross(dir, left);
	NormalizedT<OrthogonalAffineMap> result;
	result(0, 0) = left.x;  result(1, 0) = left.y;  result(2, 0) = left.z;  result(3, 0) = 0;
	result(0, 1) = newUp.x; result(1, 1) = newUp.y; result(2, 1) = newUp.z; result(3, 1) = 0;
	result(0, 2) = dir.x;   result(1, 2) = dir.y;   result(2, 2) = dir.z;   result(3, 2) = 0;
	result(0, 3) = p.x;     result(1, 3) = p.y;     result(2, 3) = p.z;     result(3, 3) = 1;
	return result;
}

float4x4 float4x4::orthographic(float clipNear, float clipFar)
{
	return Scale(Vec3f(1.0f, 1.0f, 1.0f / (clipFar - clipNear))) % Translate(Vec3f(0.0f, 0.0f, -clipNear));
}

}