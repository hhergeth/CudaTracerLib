#include "StdAfx.h"
#include "e_FPCamera.h"

bool e_FPCamera::Update()
{
	bool HASMOVED = false;
	POINT P;
	GetCursorPos(&P);
	RECT r;
	GetWindowRect(H, &r);
	//if(GetFocus() != H)
	//	return;
	if(P.x > r.right || P.x < r.left || P.y > r.bottom || P.y < r.top)
		return 0;

	float3 Velocity = make_float3(0,0,0);
	if(GetKeyState('W') < 0)
		Velocity += make_float3(0,0,Speed);
	else if (GetKeyState('S') < 0)
		Velocity -= make_float3(0,0,Speed);
	
	if (GetKeyState('A') < 0)
		Velocity -= make_float3(Speed,0,0);
	else if (GetKeyState('D') < 0)
		Velocity += make_float3(Speed,0,0);

	if(GetKeyState('E')  & 0x8000)
		Velocity -= make_float3(0,Speed,0);
	else if(GetKeyState('Q')  & 0x8000)
		Velocity += make_float3(0,Speed,0);
	HASMOVED = length(Velocity) != 0;
	m_vPos += (m_mView.Inverse()) * Velocity;

	float4x4 rot = float4x4::Identity();
	if((GetKeyState(VK_RBUTTON) & 0x80) != 0)
	{
		float xd = (r.left + (r.right - r.left) / 2.0f) - (float)P.x;
		float yd = (r.top + (r.bottom - r.top) / 2.0f) - (float)P.y;
		if(sqrtf(xd * xd + yd * yd) > 50)
		{
			P.x = r.left + (r.right - r.left) / 2;
			P.y = r.top + (r.bottom - r.top) / 2;
			SetCursorPos(P.x, P.y);
		}
		else
		{
			HASMOVED = 1;
			int w = (r.right - r.left);
			int h = (r.bottom - r.top);
			float mousemoveX = P.x - r.left - (w / 2);
			float mousemoveY = P.y - r.top - (h / 2);
			SetCursorPos(r.left + w / 2, r.top + h / 2);
			float AngleAddX = -((float)(mousemoveX / 3) * (PI / 180.0f));//thats a y rotation
			float AngleAddY = -((float)(mousemoveY / 3) * (PI / 180.0f));//thats a x rotation
			rot = float4x4::RotationAxis(m_mView.Up(), -AngleAddX) * float4x4::RotationAxis(rot.Right(), AngleAddY);
		}
	}

	m_mView = m_mView *  rot;
	return HASMOVED;
}

void e_FixedCamera::ChangeData(float px, float py, float pz, float rx, float ry, float rz, float cfov)
{
	m_bChanged = true;
	fovy *= 1.0f + cfov;
	float4x4 rot = float4x4::Identity();
	rot = float4x4::RotationAxis(m_mView.Up(), -rx) * float4x4::RotationAxis(rot.Right(), ry) * float4x4::RotationAxis(rot.Forward(), rz);
	m_mView = m_mView * rot;
	m_vPos += m_mView.Inverse() * make_float3(px, py, pz);
}

void e_FPCamera::resetZAxis()
{
	float3 r = cross(make_float3(0,-1,0), !m_mView.Z);
	float3 f = normalize(cross(!m_mView.Z, r));
	m_mView.Y = make_float4(f, 0);
}