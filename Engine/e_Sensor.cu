#include "e_Sensor.h"

void e_PerspectiveCamera::updateData()
{
	m_sampleToCamera = float4x4::Scale(make_float3(2, 2, 1)) * float4x4::Translate(make_float3(-1.0f, -1.0f, 0.0f)) * float4x4::Perspective(fov, aspect, m_fNearFarDepths.x, m_fNearFarDepths.y).Inverse();
	m_cameraToSample = m_sampleToCamera.Inverse();

	m_dx = m_sampleToCamera * make_float3(m_invResolution.x, 0.0f, 0.0f)
			- m_sampleToCamera * make_float3(0.0f);
	m_dy = m_sampleToCamera * make_float3(0.0f, m_invResolution.y, 0.0f)
			- m_sampleToCamera * make_float3(0.0f);

	float3 min = m_sampleToCamera * make_float3(0, 0, 0),
		   max = m_sampleToCamera * make_float3(1, 1, 0);
	AABB m_imageRect = AABB(min / min.z, max / max.z);
	m_normalization = 1.0f / m_imageRect.volume();
}

void e_ThinLensCamera::updateData()
{
	m_sampleToCamera = float4x4::Scale(make_float3(2, 2, 1)) * float4x4::Translate(make_float3(-1.0f, -1.0f, 0.0f)) * float4x4::Perspective(fov, aspect, m_fNearFarDepths.x, m_fNearFarDepths.y).Inverse();
	m_cameraToSample = m_sampleToCamera.Inverse();

	m_dx = m_sampleToCamera * make_float3(m_invResolution.x, 0.0f, 0.0f)
			- m_sampleToCamera * make_float3(0.0f);
	m_dy = m_sampleToCamera * make_float3(0.0f, m_invResolution.y, 0.0f)
			- m_sampleToCamera * make_float3(0.0f);

	m_aperturePdf = 1 / (PI * m_apertureRadius * m_apertureRadius);

	float3 min = m_sampleToCamera * make_float3(0, 0, 0),
		   max = m_sampleToCamera * make_float3(1, 1, 0);
	AABB m_imageRect = AABB(min / min.z, max / max.z);
	m_normalization = 1.0f / m_imageRect.volume();
}

void e_OrthographicCamera::updateData()
{
	m_sampleToCamera = float4x4::Scale(make_float3(2, 2, 1)) * float4x4::Translate(make_float3(-1.0f, -1.0f, 0.0f)) * float4x4::Orthographic(screenScale.x, screenScale.y, m_fNearFarDepths.x, m_fNearFarDepths.y).Inverse();
	m_cameraToSample = m_sampleToCamera.Inverse();

	m_invSurfaceArea = 1.0f / (
		length(toWorld * m_sampleToCamera * make_float3(1, 0, 0)) *
		length(toWorld * m_sampleToCamera * make_float3(0, 1, 0)) );
	m_scale = length(toWorld.Forward());
}

void e_TelecentricCamera::updateData()
{
	m_sampleToCamera = float4x4::Scale(make_float3(2, 2, 1)) * float4x4::Translate(make_float3(-1.0f, -1.0f, 0.0f)) * float4x4::Orthographic(screenScale.x, screenScale.y, m_fNearFarDepths.x, m_fNearFarDepths.y).Inverse();
	m_cameraToSample = m_sampleToCamera.Inverse();

	m_normalization = 1.0f / (
		length(toWorld * m_sampleToCamera * make_float3(1, 0, 0)) *
		length(toWorld * m_sampleToCamera * make_float3(0, 1, 0)) );

	m_aperturePdf = 1.0f / (PI * m_apertureRadius * m_apertureRadius);
}

float4x4 e_Sensor::View() const
{
	float4x4 m_mView = As<e_SensorBase>()->getWorld();
	float3 pos = m_mView.Translation();
	m_mView = m_mView * float4x4::Translate(-pos);
	return m_mView;
}

float3 e_Sensor::Position() const
{
	return As<e_SensorBase>()->getWorld().Translation();
}

void e_Sensor::SetToWorld(const float3& pos, const float4x4& rot)
{
	SetToWorld(rot * float4x4::Translate(pos));
}

void e_Sensor::SetToWorld(const float3& pos, const float3& f)
{
	float3 r = cross(f, make_float3(0,1,0));
	float3 u = cross(r, f);
	float4x4 m_mView = float4x4::Identity();
	m_mView.X = make_float4(r.x, r.y, r.z, 0.0f);
	m_mView.Y = make_float4(u.x, u.y, u.z, 0.0f);
	m_mView.Z = make_float4(f.x, f.y, f.z, 0.0f);
	SetToWorld(pos, m_mView);
}

void e_Sensor::SetFilmData(int w, int h)
{
	As<e_SensorBase>()->SetFilmData(w, h);
}

void e_Sensor::SetToWorld(const float4x4& w)
{
	As()->SetToWorld(w);
}

float4x4 e_Sensor::getGLViewProjection() const
{
	float4x4 proj = getProjectionMatrix();
	return (float4x4::Translate(-1.0f * Position()) * (View().Inverse() )) * proj;
}