#pragma once

#include "..\Math\vector.h"

struct e_CameraData
{
	float4x4 view;
	float4x4 proj;
	float3 p;
	float dist;
	float apperture;
	float4x4 m_mViewProj;
	CUDA_FUNC e_CameraData(){}
	CUDA_ONLY_FUNC Ray GenRay(int2 pos, int2 size) const
	{
		return GenRay(pos.x, pos.y, size.x, size.y);
	}
	CUDA_ONLY_FUNC Ray GenRay(float x, float y, float w, float h) const
	{
		float4 a = make_float4(2.0f * ((float)x / (float)w) - 1.0f, -(2.0f * ((float)y / (float)h) - 1.0f), 0, 1.0f);
		a = proj * a;
		a /= a.w;
		return Ray(p, normalize(view.TransformNormal(!a)));
	}
	CUDA_ONLY_FUNC Ray GenRay(int2 pos, int2 size, float u, float v) const
	{
		return GenRay(pos.x, pos.y, size.x, size.y, u, v);
	}
	CUDA_ONLY_FUNC Ray GenRay(float x, float y, float w, float h, float u, float v) const
	{
		float4 a = make_float4(2.0f * ((float)x / (float)w) - 1.0f, -(2.0f * ((float)y / (float)h) - 1.0f), 0, 1.0f);
		a = proj * a;
		a /= a.w;
		float3 b = normalize(!a);
		float3 tar = b * dist, p0 = make_float3(u,v,0) * apperture, d = tar - p0;
		return Ray(view * p0, normalize(view.TransformNormal(d)));
	}
};

class e_Camera
{
protected:
	float m_fDist;
	float aspect;
	float apperture;
public:
	e_Camera()
	{
	}
	virtual bool Update() = 0;
	virtual void getData(e_CameraData& c) const  = 0;
	virtual float4x4 getViewProjection() const = 0;
	//virtual void getOpenGLData(float* view, float* proj) = 0;
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
