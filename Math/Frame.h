#pragma once

#include "float4x4.h"

CUDA_FUNC_IN void coordinateSystem(const float3& a, float3& s, float3& t)
{
	if (abs(a.x) > abs(a.y))
	{
		float invLen = 1.0f / sqrtf(a.x * a.x + a.z * a.z);
		t = make_float3(a.z * invLen, 0.0f, -a.x * invLen);
	}
	else
	{
		float invLen = 1.0f / sqrtf(a.y * a.y + a.z * a.z);
		t = make_float3(0.0f, a.z * invLen, -a.y * invLen);
	}
	s = normalize(cross(t, a));
}

struct Frame
{
	float3 s, t;
	float3 n;

	/// Default constructor -- performs no initialization!
	CUDA_FUNC_IN Frame() { }

	/// Given a float3 and tangent float3s, construct a new coordinate frame
	CUDA_FUNC_IN Frame(const float3 &s, const float3 &t, const float3 &n)
	 : s(s), t(t), n(n) {
	}

	/// Construct a new coordinate frame from a single float3
	CUDA_FUNC_IN Frame(const float3 &n) : n(normalize(n)) {
		coordinateSystem(n, s, t);
	}

	/// Convert from world coordinates to local coordinates
	CUDA_FUNC_IN float3 toLocal(const float3 &v) const {
		return make_float3(
			dot(v, s),
			dot(v, t),
			dot(v, n)
		);
	}

	/// Convert from local coordinates to world coordinates
	CUDA_FUNC_IN float3 toWorld(const float3 &v) const {
		return s * v.x + t * v.y + n * v.z;
	}

	CUDA_FUNC_IN float4x4 ToMatrix()
	{
		float4x4 r;
		r.col(0, make_float4(t, 0));
		r.col(1, make_float4(s, 0));
		r.col(2, make_float4(n, 0));
		r.col(3, make_float4(0, 0, 0, 1));
		return r;
	}

	/** \brief Assuming that the given direction is in the local coordinate
	 * system, return the squared cosine of the angle between the float3 and v */
	CUDA_FUNC_IN static float cosTheta2(const float3 &v) {
		return v.z * v.z;
	}

	/** \brief Assuming that the given direction is in the local coordinate
	 * system, return the cosine of the angle between the float3 and v */
	CUDA_FUNC_IN static float cosTheta(const float3 &v) {
		return v.z;
	}

	/** \brief Assuming that the given direction is in the local coordinate
	 * system, return the squared sine of the angle between the float3 and v */
	CUDA_FUNC_IN static float sinTheta2(const float3 &v) {
		return 1.0f - v.z * v.z;
	}

	/** \brief Assuming that the given direction is in the local coordinate
	 * system, return the sine of the angle between the float3 and v */
	CUDA_FUNC_IN static float sinTheta(const float3 &v) {
		float temp = sinTheta2(v);
		if (temp <= 0.0f)
			return 0.0f;
		return sqrtf(temp);
	}

	/** \brief Assuming that the given direction is in the local coordinate
	 * system, return the tangent of the angle between the float3 and v */
	CUDA_FUNC_IN static float tanTheta(const float3 &v) {
		float temp = 1 - v.z*v.z;
		if (temp <= 0.0f)
			return 0.0f;
		return sqrtf(temp) / v.z;
	}

	/** \brief Assuming that the given direction is in the local coordinate
	 * system, return the squared tangent of the angle between the float3 and v */
	CUDA_FUNC_IN static float tanTheta2(const float3 &v) {
		float temp = 1 - v.z*v.z;
		if (temp <= 0.0f)
			return 0.0f;
		return temp / (v.z * v.z);
	}

	/** \brief Assuming that the given direction is in the local coordinate
	 * system, return the sine of the phi parameter in spherical coordinates */
	CUDA_FUNC_IN static float sinPhi(const float3 &v) {
		float sinTheta = Frame::sinTheta(v);
		if (sinTheta == 0.0f)
			return 1.0f;
		return clamp(v.y / sinTheta, (float) -1.0f, (float) 1.0f);
	}

	/** \brief Assuming that the given direction is in the local coordinate
	 * system, return the cosine of the phi parameter in spherical coordinates */
	CUDA_FUNC_IN static float cosPhi(const float3 &v) {
		float sinTheta = Frame::sinTheta(v);
		if (sinTheta == 0.0f)
			return 1.0f;
		return clamp(v.x / sinTheta, (float) -1.0f, (float) 1.0f);
	}

	/** \brief Assuming that the given direction is in the local coordinate
	 * system, return the squared sine of the phi parameter in  spherical
	 * coordinates */
	CUDA_FUNC_IN static float sinPhi2(const float3 &v) {
		return clamp(v.y * v.y / sinTheta2(v), (float) 0.0f, (float) 1.0f);
	}

	/** \brief Assuming that the given direction is in the local coordinate
	 * system, return the squared cosine of the phi parameter in  spherical
	 * coordinates */
	CUDA_FUNC_IN static float cosPhi2(const float3 &v) {
		return clamp(v.x * v.x / sinTheta2(v), (float) 0.0f, (float) 1.0f);
	}

	CUDA_FUNC_IN Frame operator *(const float4x4& m) const
	{
		Frame r;
		r.s = normalize(m.TransformDirection(s));
		r.t = normalize(m.TransformDirection(t));
		r.n = normalize(cross(r.t, r.s));
		return r;
	}

	/// Equality test
	CUDA_FUNC_IN bool operator==(const Frame &frame) const {
		float3 diff = frame.s - s + frame.t - t + frame.n - n;
		return !dot(diff, diff);
	}

	/// Inequality test
	CUDA_FUNC_IN bool operator!=(const Frame &frame) const {
		return !operator==(frame);
	}
};