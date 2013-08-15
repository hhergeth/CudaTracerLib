#pragma once

#include "..\Math\vector.h"
#include "..\Math\AABB.h"

struct e_CameraData
{
	float4x4 view;
	float4x4 proj;
	float3 p;
	float dist;
	float apperture;
	float4x4 m_mViewProj;
	CUDA_FUNC e_CameraData(){}

	CUDA_ONLY_FUNC Ray GenRay(CameraSample& sample, int2 size) const
	{
		return GenRay(sample, size.x, size.y);
	}

	CUDA_ONLY_FUNC Ray GenRay(CameraSample& sample, float w, float h) const
	{
		float4 a = make_float4(2.0f * (sample.imageX / (float)w) - 1.0f, -(2.0f * (sample.imageY / (float)h) - 1.0f), 0, 1.0f);
		a = proj * a;
		a /= a.w;
		float3 b = normalize(!a);
		float3 tar = b * dist, p0 = make_float3(sample.lensU,sample.lensV,0) * apperture, d = tar - p0;
		return Ray(view * p0, normalize(view.TransformNormal(d)));
	}

	CUDA_FUNC_IN Ray GenRay(int2 _p, int2 size) const
	{
		float4 a = make_float4(2.0f * ((float)_p.x / (float)size.x) - 1.0f, -(2.0f * ((float)_p.y / (float)size.y) - 1.0f), 0, 1.0f);
		a = proj * a;
		a /= a.w;
		return Ray(p, normalize(view.TransformNormal(!a)));
	}
};

class e_Camera
{
protected:
	float m_fDist;
	float aspect;
	float apperture;
public:
	AABB m_sLastFrustum;
	float2 m_fNearFarDepths;
public:
	e_Camera(float2 depths = make_float2(1, 1000))
		: m_fNearFarDepths(depths)
	{
	}
	virtual bool Update() = 0;
	virtual e_CameraData getData() const  = 0;
	virtual float4x4 getGLViewProjection() const = 0;
	float3 getPos()
	{
		return getData().p;
	}
	float3 getDir()
	{
		return getData().view.Forward();
	}
	virtual void Set(float3& pos, float3& tar) = 0;
	virtual void Set(float3& pos, float3& tar, float3& up) = 0;
	virtual void setFocalDepth(float f)
	{
		m_fDist = f;
	}
	virtual float& accessApperture()
	{
		return apperture;
	}
	virtual void speedChange(float dir)
	{

	}
	virtual void setSpeed(float speed)
	{

	}
	virtual void UpdateAfterResize(int w, int h) = 0;
};
