#pragma once

#ifndef ISWINDOWS
#pragma message("Can't use camera controllers without msvcpp")
#endif

#include "e_Sensor.h"

class e_CameraController
{
	HWND H;
public:
	float Speed;
	e_Sensor* Camera;

	e_CameraController(HWND h, e_Sensor* S, float _speed = 1.0f)
	{
		Camera = S;
		H = h;
		Speed = _speed;
	}

	///returns wether the camera has been changed
	bool Update()
	{
		if(GetFocus() != H)
			return false;

		bool HASMOVED = false;
		POINT P;
		GetCursorPos(&P);
		RECT rect;
		GetWindowRect(H, &rect);
		//if(GetFocus() != H)
		//	return;
		if(P.x > rect.right || P.x < rect.left || P.y > rect.bottom || P.y < rect.top)
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
			Velocity += make_float3(0,Speed,0);
		else if(GetKeyState('Q')  & 0x8000)
			Velocity -= make_float3(0,Speed,0);
		HASMOVED = length(Velocity) != 0;
		float4x4 m_mView = Camera->View();
		float3 pos = Camera->Position() + m_mView.TransformNormal(Velocity);

		if((GetKeyState(VK_RBUTTON) & 0x80) != 0)
		{
			float xd = (rect.left + (rect.right - rect.left) / 2.0f) - (float)P.x;
			float yd = (rect.top + (rect.bottom - rect.top) / 2.0f) - (float)P.y;
			if(sqrtf(xd * xd + yd * yd) > 50)
			{
				P.x = rect.left + (rect.right - rect.left) / 2;
				P.y = rect.top + (rect.bottom - rect.top) / 2;
				SetCursorPos(P.x, P.y);
			}
			else
			{
				HASMOVED = 1;
				int w = (rect.right - rect.left);
				int h = (rect.bottom - rect.top);
				float mousemoveX = P.x - rect.left - (w / 2);
				float mousemoveY = P.y - rect.top - (h / 2);
				SetCursorPos(rect.left + w / 2, rect.top + h / 2);
				float AngleAddX = -((float)(mousemoveX / 3) * (PI / 180.0f));//moving the mouse from left to right -> rotation around Y axis
				float AngleAddY = -((float)(mousemoveY / 3) * (PI / 180.0f));//thats a x rotation
				m_mView *= float4x4::RotationAxis(m_mView.Right(), -AngleAddY);
				m_mView *= float4x4::RotationAxis(m_mView.Up(), AngleAddX);
			}
		}

		float3 f = normalize(!m_mView.Z), r = normalize(cross(f, make_float3(0,1,0))), u = normalize(cross(r, f));
		m_mView.X = make_float4(r.x, r.y, r.z, 0.0f);
		m_mView.Y = make_float4(u.x, u.y, u.z, 0.0f);
		m_mView.Z = make_float4(f.x, f.y, f.z, 0.0f);

		Camera->SetToWorld(pos, m_mView);
		return HASMOVED;
	}
};