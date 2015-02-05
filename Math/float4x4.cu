#include "float4x4.h"

float4x4 float4x4::inverse() const
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
