#pragma once

#include "e_Camera.h"

/*
	virtual void getOpenGLData(float* view, float* proj)
	{
		*(float4x4*)proj = float4x4::Perspective(fovy, aspect, 1, 1000);
		*(float4x4*)view = (float4x4::Translate(-m_vPos)*(m_mView * float4x4(1,0,0,0,0,-1,0,0,0,0,1,0,0,0,0,1)));
	}*/

class e_FPCamera : public e_Camera
{
private:
	float fovy;
	float Speed;
	float4x4 m_mView;
	float3 m_vPos;
	HWND H;
public:
	e_FPCamera(HWND h, float _fov, float _speed)
	{
		H = h;
		apperture = 0.0f;
		m_fDist = 1;
		m_mView = float4x4::Identity() * float4x4::RotateZ(PI);
		m_vPos = make_float3(0);
		this->fovy = _fov;
		Speed = _speed;
		RECT r;
		GetWindowRect(H, &r);
		int w = (r.right - r.left);
		int h2 = (r.bottom - r.top);
		UpdateAfterResize(w, h2);
	}
	virtual void UpdateAfterResize(int w, int h)
	{
		aspect = (float)w / (float)h;
	}
	virtual e_CameraData getData() const
	{
		e_CameraData c;
		c.p = m_vPos;
		float4x4 p = getProj(), v = float4x4::Translate(-1.0f * m_vPos) * m_mView;
		c.proj = p.Inverse();
		c.view = v.Inverse();
		c.m_mViewProj = v * p;
		c.dist = m_fDist;
		c.apperture = apperture;
		return c;
	}
	virtual float4x4 getGLViewProjection() const
	{
		return (float4x4::Translate(-1.0f * m_vPos) * (m_mView * float4x4(1,0,0,0,0,-1,0,0,0,0,1,0,0,0,0,1))) * getProj();
	}
	virtual bool Update();
	virtual void Set(float3& pos, float3& tar)
	{
		m_vPos = pos;
		float3 up = make_float3(0,1,0), f = normalize(tar - pos), r = normalize(cross(f, up)), u = normalize(cross(f, r));
		m_mView.X = make_float4(r.x, r.y, r.z, 0.0f);
		m_mView.Y = make_float4(u.x, u.y, u.z, 0.0f);
		m_mView.Z = make_float4(f.x, f.y, f.z, 0.0f);
	}
	virtual void Set(float3& pos, float3& tar, float3& up)
	{
		m_vPos = pos;
		float3 f = normalize(tar - pos), r = normalize(cross(f, up)), u = normalize(up);
		m_mView.X = make_float4(r.x, r.y, r.z, 0.0f);
		m_mView.Y = make_float4(u.x, u.y, u.z, 0.0f);
		m_mView.Z = make_float4(f.x, f.y, f.z, 0.0f);
	}
	virtual void speedChange(float dir)
	{
		float e = 0.1f;
		Speed *= 1 + dir * e;
	}
	virtual void setSpeed(float speed)
	{
		Speed = speed;
	}
	float4x4 getProj() const
	{
		return float4x4::Perspective(fovy, aspect, m_fNearFarDepths.x, m_fNearFarDepths.y);
	}
};

class e_FixedCamera : public e_Camera
{
private:
	float fovy;
	float4x4 m_mView;
	float3 m_vPos;
	bool m_bChanged;
public:
	e_FixedCamera(float _fov, int w, int h)
	{
		m_bChanged = true;
		apperture = 0.0f;
		m_fDist = 1;
		m_mView = float4x4::Identity() * float4x4::RotateZ(PI);
		m_vPos = make_float3(0);
		this->fovy = _fov;
		UpdateAfterResize(w, h);
	}
	virtual bool Update()
	{
		bool a = m_bChanged;
		m_bChanged = false;
		return a;
	}
	void UpdateAfterResize(int w, int h)
	{
		aspect = (float)w / (float)h;
		m_bChanged = true;
	}
	virtual void getData(e_CameraData& c) const
	{
		c.p = m_vPos;
		float4x4 p = getProj(), v = float4x4::Translate(-1.0f * m_vPos) * m_mView;
		c.proj = p.Inverse();
		c.view = v.Inverse();
		c.m_mViewProj = v * p;
		c.dist = m_fDist;
		c.apperture = apperture;
	}
	virtual float4x4 getViewProjection() const
	{
		return (float4x4::Translate(-1.0f * m_vPos) * (m_mView * float4x4(1,0,0,0,0,-1,0,0,0,0,1,0,0,0,0,1))) * getProj();
	}
	virtual void Set(float3& pos, float3& tar)
	{
		m_bChanged = true;
		m_vPos = pos;
		float3 up = make_float3(0,1,0), f = normalize(tar - pos), r = normalize(cross(f, up)), u = normalize(cross(f, r));
		m_mView.X = make_float4(r.x, r.y, r.z, 0.0f);
		m_mView.Y = make_float4(u.x, u.y, u.z, 0.0f);
		m_mView.Z = make_float4(f.x, f.y, f.z, 0.0f);
	}
	void SetData(float x, float y, float z, float dx, float dy, float dz, float fov)
	{
		float3 p = make_float3(x,y,z);
		Set(p, p + make_float3(dx,dy,dz));
		fovy = fov;
	}
	void ChangeData(float px, float py, float pz, float rx, float ry, float rz, float cfov);
	virtual void Set(float3& pos, float3& tar, float3& up)
	{
		m_bChanged = true;
		m_vPos = pos;
		float3 f = normalize(tar - pos), r = normalize(cross(f, up)), u = normalize(up);
		m_mView.X = make_float4(r.x, r.y, r.z, 0.0f);
		m_mView.Y = make_float4(u.x, u.y, u.z, 0.0f);
		m_mView.Z = make_float4(f.x, f.y, f.z, 0.0f);
	}
	virtual void speedChange(float dir)
	{
	}
	virtual void setSpeed(float speed)
	{
	}
	float4x4 getProj() const
	{
		return float4x4::Perspective(fovy, aspect, 1, 1000);
	}
};