#pragma once

#include "cutil_math.h"

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
};