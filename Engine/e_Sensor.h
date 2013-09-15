#pragma once

#include "..\MathTypes.h"
#include "e_Samples.h"

//this architecture and the implementations are completly copied from mitsuba!

struct e_SensorBase
{
protected:
	float aspect;
	float2 m_resolution, m_invResolution;
	float4x4 toWorld;
	float2 m_fNearFarDepths;
	float fov;
	float m_apertureRadius;
	float m_focusDistance;
protected:
	virtual void updateData()
	{
	}
public:
	const bool isFieldOfViewSensor;
	const bool isAppertureSensor;
	e_SensorBase(bool fov, bool app)
		: isFieldOfViewSensor(fov), isAppertureSensor(app)
	{
		toWorld = float4x4::Identity();
	}
	virtual void SetToWorld(const float4x4& w)
	{
		toWorld = w;
	}	
	virtual void SetFov(float _fov)
	{
		fov = _fov;
		updateData();
	}
	virtual void SetNearFarDepth(float nearD, float farD)
	{
		m_fNearFarDepths = make_float2(nearD, farD);
		updateData();
	}
	virtual void SetFilmData(int w, int h)
	{
		m_resolution = make_float2(w, h);
		m_invResolution = make_float2(1) / m_resolution;
		aspect = m_resolution.x / m_resolution.y;
		updateData();
	}
	virtual void SetApperture(float a)
	{
		m_apertureRadius = a;
		updateData();
	}
	virtual void SetFocalDistance(float a)
	{
		m_focusDistance = a;
		updateData();
	}
	float4x4 getWorld()
	{
		return toWorld;
	}
};

#define e_SphericalCamera_TYPE 1
struct e_SphericalCamera : public e_SensorBase
{
	e_SphericalCamera(int w, int h)
		: e_SensorBase(false, false)
	{
		SetFilmData(w, h);
	}

	CUDA_FUNC_IN Spectrum sampleRay(Ray &ray, const float2 &pixelSample, const float2 &apertureSample) const
	{
		float sinPhi, cosPhi, sinTheta, cosTheta;
		sincos((1.0f - pixelSample.x * m_invResolution.x) * 2 * PI, &sinPhi, &cosPhi);
		sincos((1.0f - pixelSample.y * m_invResolution.y) * PI, &sinTheta, &cosTheta);

		float3 d = make_float3(sinPhi*sinTheta, cosTheta, -cosPhi*sinTheta);
		ray = Ray(toWorld.Translation(), toWorld.TransformNormal(d));

		return Spectrum(1.0f);
	}

	CUDA_FUNC_IN Spectrum eval(const Ray& r, const TraceResult& r2, const float3 &d, float2 &samplePos) const
	{
		return Spectrum(1.0f);
	}

	CUDA_FUNC_IN Spectrum samplePosition(PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const
	{
		pRec.p = toWorld.Translation();
		pRec.n = make_float3(0.0f);
		pRec.pdf = 1.0f;
		pRec.measure = EDiscrete;
		return Spectrum(1.0f);
	}

	CUDA_FUNC_IN Spectrum evalPosition(const PositionSamplingRecord &pRec) const
	{
		return Spectrum((pRec.measure == EDiscrete) ? 1.0f : 0.0f);
	}

	CUDA_FUNC_IN float pdfPosition(const PositionSamplingRecord &pRec) const
	{
		return (pRec.measure == EDiscrete) ? 1.0f : 0.0f;
	}

	CUDA_FUNC_IN Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const
	{
		float3 samplePos = make_float3(sample.x, sample.y, 0.0f);

		if (extra)
		{
			/* The caller wants to condition on a specific pixel position */
			samplePos.x = (extra->x + sample.x) * m_invResolution.x;
			samplePos.y = (extra->y + sample.y) * m_invResolution.y;
		}

		pRec.uv = make_float2(samplePos.x * m_resolution.x, samplePos.y * m_resolution.y);

		float sinPhi, cosPhi, sinTheta, cosTheta;
		sincos(samplePos.x * 2 * PI, &sinPhi, &cosPhi);
		sincos(samplePos.y * PI, &sinTheta, &cosTheta);

		dRec.d = toWorld * make_float3(sinPhi*sinTheta, cosTheta, -cosPhi*sinTheta);
		dRec.measure = ESolidAngle;
		dRec.pdf = 1 / (2 * PI * PI * MAX(sinTheta, EPSILON));

		return Spectrum(1.0f);
	}

	CUDA_FUNC_IN float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		if (dRec.measure != ESolidAngle)
			return 0.0f;

		float3 d = toWorld.Inverse().TransformNormal(dRec.d);
		float sinTheta = math::safe_sqrt(1-d.y*d.y);

		return 1 / (2 * PI * PI * MAX(sinTheta, EPSILON));
	}

	CUDA_FUNC_IN Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		if (dRec.measure != ESolidAngle)
			return Spectrum(0.0f);

		float3 d = toWorld.Inverse().TransformNormal(dRec.d);
		float sinTheta = math::safe_sqrt(1-d.y*d.y);

		return Spectrum(1 / (2 * PI * PI * MAX(sinTheta, EPSILON)));
	}

	CUDA_FUNC_IN bool getSamplePosition(const PositionSamplingRecord &pRec, const DirectionSamplingRecord &dRec, float2 &samplePosition) const
	{
		float3 d = normalize(toWorld.Inverse().TransformNormal(dRec.d));

		samplePosition = make_float2(
			math::modulo(atan2(d.x, -d.z) * INV_TWOPI, (float) 1) * m_resolution.x,
			math::safe_acos(d.y) * INV_PI * m_resolution.y
		);

		return true;
	}

	float4x4 getProjectionMatrix() const
	{
		return float4x4::Identity();
	}

	TYPE_FUNC(e_SphericalCamera)
};

#define e_PerspectiveCamera_TYPE 2
struct e_PerspectiveCamera : public e_SensorBase
{
protected:
	virtual void updateData();
	float4x4 m_cameraToSample, m_sampleToCamera;
	float3 m_dx, m_dy;
	float m_normalization;
public:
	e_PerspectiveCamera(int w, int h, float _fov)
		: e_SensorBase(true, false)
	{
		m_fNearFarDepths = make_float2(1, 100000);
		SetFilmData(w, h);
		fov = _fov;
		updateData();
	}

	CUDA_FUNC_IN float importance(const float3 &d) const
	{
		float cosTheta = Frame::cosTheta(d);

		/* Check if the direction points behind the camera */
		if (cosTheta <= 0)
			return 0.0f;

		/* Compute the position on the plane at distance 1 */
		float invCosTheta = 1.0f / cosTheta;
		float2 p = make_float2(d.x * invCosTheta, d.y * invCosTheta);

		return invCosTheta * invCosTheta * invCosTheta * m_normalization;
	}

	CUDA_FUNC_IN Spectrum sampleRay(Ray &ray, const float2 &pixelSample, const float2 &apertureSample) const
	{
		float3 nearP = m_sampleToCamera * make_float3(
			pixelSample.x * m_invResolution.x,
			pixelSample.y * m_invResolution.y, 0.0f);

		/* Turn that into a normalized ray direction, and
		   adjust the ray interval accordingly */
		float3 d = normalize(nearP);
		ray = Ray(toWorld.Translation(), toWorld.TransformNormal(d));

		return Spectrum(1.0f);
	}

	CUDA_FUNC_IN Spectrum eval(const Ray& r, const TraceResult& r2, const float3 &d, float2 &samplePos) const
	{
		return Spectrum(1.0f);
	}

	CUDA_FUNC_IN Spectrum samplePosition(PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const
	{
		pRec.p = toWorld.Translation();
		pRec.n = toWorld.Forward();
		pRec.pdf = 1.0f;
		pRec.measure = EDiscrete;
		return Spectrum(1.0f);
	}

	CUDA_FUNC_IN Spectrum evalPosition(const PositionSamplingRecord &pRec) const
	{
		return Spectrum((pRec.measure == EDiscrete) ? 1.0f : 0.0f);
	}

	CUDA_FUNC_IN float pdfPosition(const PositionSamplingRecord &pRec) const
	{
		return (pRec.measure == EDiscrete) ? 1.0f : 0.0f;
	}

	CUDA_FUNC_IN Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const
	{
		float3 samplePos = make_float3(sample.x, sample.y, 0.0f);

		if (extra) {
			/* The caller wants to condition on a specific pixel position */
			samplePos.x = (extra->x + sample.x) * m_invResolution.x;
			samplePos.y = (extra->y + sample.y) * m_invResolution.y;
		}

		pRec.uv = make_float2(samplePos.x * m_resolution.x,
			samplePos.y * m_resolution.y);

		/* Compute the corresponding position on the
		   near plane (in local camera space) */
		float3 nearP = m_sampleToCamera * samplePos;

		/* Turn that into a normalized ray direction */
		float3 d = normalize(nearP);
		dRec.d = toWorld.TransformNormal(d);
		dRec.measure = ESolidAngle;
		dRec.pdf = m_normalization / (d.z * d.z * d.z);

		return Spectrum(1.0f);
	}

	CUDA_FUNC_IN float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		if (dRec.measure != ESolidAngle)
			return 0.0f;

		return importance(toWorld.Inverse().TransformNormal(dRec.d));
	}

	CUDA_FUNC_IN Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		if (dRec.measure != ESolidAngle)
			return Spectrum(0.0f);

		return Spectrum(importance(toWorld.Inverse().TransformNormal(dRec.d)));
	}

	CUDA_FUNC_IN bool getSamplePosition(const PositionSamplingRecord &pRec, const DirectionSamplingRecord &dRec, float2 &samplePosition) const
	{
		float3 local = toWorld.Inverse().TransformNormal(dRec.d);

		if (local.z <= 0)
			return false;

		float3 screenSample = m_cameraToSample * local;
		if (screenSample.x < 0 || screenSample.x > 1 ||
			screenSample.y < 0 || screenSample.y > 1)
			return false;

		samplePosition = make_float2(
				screenSample.x * m_resolution.x,
				screenSample.y * m_resolution.y);

		return true;
	}

	float4x4 getProjectionMatrix() const
	{
		return float4x4::Perspective(fov, aspect, m_fNearFarDepths.x, m_fNearFarDepths.y);
	}

	TYPE_FUNC(e_PerspectiveCamera)
};

#define e_ThinLensCamera_TYPE 3
struct e_ThinLensCamera : public e_SensorBase
{
protected:
	virtual void updateData();
	float4x4 m_cameraToSample, m_sampleToCamera;
	float3 m_dx, m_dy;
	float m_aperturePdf;
	float m_normalization;
	CUDA_FUNC_IN float importance(const float3 &p, const float3 &d, float2* sample = 0) const
	{
		float cosTheta = Frame::cosTheta(d);
		if (cosTheta <= 0)
			return 0.0f;
		float invCosTheta = 1.0f / cosTheta;
		float3 scr = m_cameraToSample * (p + d * (m_focusDistance*invCosTheta));
		if (scr.x < 0 || scr.x > 1 ||
			scr.y < 0 || scr.y > 1)
			return 0.0f;

		if (sample) {
			sample->x = scr.x * m_resolution.x;
			sample->y = scr.y * m_resolution.y;
		}

		return m_normalization * invCosTheta *
			invCosTheta * invCosTheta;
	}
public:
	e_ThinLensCamera(int w, int h, float _fov, float a, float dist)
		: e_SensorBase(true, true)
	{
		m_fNearFarDepths = make_float2(1, 100000);
		SetFilmData(w, h);
		fov = _fov;
		m_apertureRadius = a;
		m_focusDistance = dist;
		updateData();
	}

	CUDA_FUNC_IN Spectrum sampleRay(Ray &ray, const float2 &pixelSample, const float2 &apertureSample) const
	{
		float2 tmp = Warp::squareToUniformDiskConcentric(apertureSample) * m_apertureRadius;

		/* Compute the corresponding position on the
		   near plane (in local camera space) */
		float3 nearP = m_sampleToCamera * make_float3(
			pixelSample.x * m_invResolution.x,
			pixelSample.y * m_invResolution.y, 0.0f);

		/* Aperture position */
		float3 apertureP = make_float3(tmp.x, tmp.y, 0.0f);

		/* Sampled position on the focal plane */
		float3 focusP = nearP * (m_focusDistance / nearP.z);

		/* Turn these into a normalized ray direction, and
		   adjust the ray interval accordingly */
		float3 d = normalize(focusP - apertureP);
		
		ray = Ray(toWorld * apertureP, toWorld.TransformNormal(d));

		return Spectrum(1.0f);
	}

	CUDA_FUNC_IN Spectrum eval(const Ray& r, const TraceResult& r2, const float3 &d, float2 &samplePos) const
	{
		return Spectrum(0.0f);
	}

	CUDA_FUNC_IN Spectrum samplePosition(PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const
	{
		float2 aperturePos = Warp::squareToUniformDiskConcentric(sample) * m_apertureRadius;

		pRec.p = toWorld * make_float3(aperturePos.x, aperturePos.y, 0.0f);
		pRec.n = toWorld.Forward();
		pRec.pdf = m_aperturePdf;
		pRec.measure = EArea;
		return Spectrum(1.0f);
	}

	CUDA_FUNC_IN Spectrum evalPosition(const PositionSamplingRecord &pRec) const
	{
		return Spectrum((pRec.measure == EArea) ? m_aperturePdf : 0.0f);
	}

	CUDA_FUNC_IN float pdfPosition(const PositionSamplingRecord &pRec) const
	{
		return (pRec.measure == EArea) ? m_aperturePdf : 0.0f;
	}

	CUDA_FUNC_IN Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const
	{
		float3 samplePos = make_float3(sample.x, sample.y, 0.0f);

		if (extra) {
			/* The caller wants to condition on a specific pixel position */
			samplePos.x = (extra->x + sample.x) * m_invResolution.x;
			samplePos.y = (extra->y + sample.y) * m_invResolution.y;
		}

		pRec.uv = make_float2(samplePos.x * m_resolution.x,
			samplePos.y * m_resolution.y);

		/* Compute the corresponding position on the
		   near plane (in local camera space) */
		float3 nearP = m_sampleToCamera * samplePos;
		nearP.x = nearP.x * (m_focusDistance / nearP.z);
		nearP.y = nearP.y * (m_focusDistance / nearP.z);
		nearP.z = m_focusDistance;

		float3 apertureP = toWorld.Inverse() * pRec.p;

		/* Turn that into a normalized ray direction */
		float3 d = normalize(nearP - apertureP);
		dRec.d = toWorld.TransformNormal(d);
		dRec.measure = ESolidAngle;
		dRec.pdf = m_normalization / (d.z * d.z * d.z);

		return Spectrum(1.0f);
	}

	CUDA_FUNC_IN float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		if (dRec.measure != ESolidAngle)
			return 0.0f;

		float4x4 invToWorld = toWorld.Inverse();
		return importance(invToWorld * pRec.p, invToWorld.TransformNormal(dRec.d));
	}

	CUDA_FUNC_IN Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		if (dRec.measure != ESolidAngle)
			return Spectrum(0.0f);

		float4x4 invToWorld = toWorld.Inverse();
		return Spectrum(importance(invToWorld * pRec.p, invToWorld.TransformNormal(dRec.d)));
	}

	CUDA_FUNC_IN bool getSamplePosition(const PositionSamplingRecord &pRec, const DirectionSamplingRecord &dRec, float2 &samplePosition) const
	{
		float4x4 invToWorld = toWorld.Inverse();
		float3 localP(invToWorld * pRec.p);
		float3 localD(invToWorld.TransformNormal(dRec.d));

		if (localD.z <= 0)
			return false;

		float3 intersection = localP + localD * (m_focusDistance / localD.z);

		float3 screenSample = m_cameraToSample * intersection;
		if (screenSample.x < 0 || screenSample.x > 1 ||
			screenSample.y < 0 || screenSample.y > 1)
			return false;

		samplePosition = make_float2(
				screenSample.x * m_resolution.x,
				screenSample.y * m_resolution.y);

		return true;
	}

	float4x4 getProjectionMatrix() const
	{
		return float4x4::Perspective(fov, aspect, m_fNearFarDepths.x, m_fNearFarDepths.y);
	}

	TYPE_FUNC(e_ThinLensCamera)
};

#define e_OrthographicCamera_TYPE 4
struct e_OrthographicCamera : public e_SensorBase
{
protected:
	virtual void updateData();
	float4x4 m_cameraToSample;
	float4x4 m_sampleToCamera;
	float m_invSurfaceArea, m_scale;
	float2 screenScale;
public:
	e_OrthographicCamera(int w, int h, float sx = 1, float sy = 1)
		: e_SensorBase(false, false)
	{
		m_fNearFarDepths = make_float2(0.00001f, 100000);
		SetFilmData(w, h);
		SetScreenScale(sx, sy);
	}

	void SetScreenScale(float sx, float sy)
	{
		screenScale = make_float2(sx, sy);
		updateData();
	}

	CUDA_FUNC_IN Spectrum sampleRay(Ray &ray, const float2 &pixelSample, const float2 &apertureSample) const
	{
		float3 nearP = m_sampleToCamera * make_float3(
			pixelSample.x * m_invResolution.x,
			pixelSample.y * m_invResolution.y, 0.0f);

		ray = Ray(toWorld * make_float3(nearP.x, nearP.y, 0.0f), toWorld.Forward());

		return Spectrum(1.0f);
	}

	CUDA_FUNC_IN Spectrum eval(const Ray& r, const TraceResult& r2, const float3 &d, float2 &samplePos) const
	{
		return Spectrum(0.0f);
	}

	CUDA_FUNC_IN Spectrum samplePosition(PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const
	{
		float3 samplePos = make_float3(sample.x, sample.y, 0.0f);

		if (extra) {
			/* The caller wants to condition on a specific pixel position */
			samplePos.x = (extra->x + sample.x) * m_invResolution.x;
			samplePos.y = (extra->y + sample.y) * m_invResolution.y;
		}

		pRec.uv = make_float2(samplePos.x * m_resolution.x,	samplePos.y * m_resolution.y);

		float3 nearP = m_sampleToCamera * samplePos;

		nearP.z = 0.0f;
		pRec.p = toWorld * nearP;
		pRec.n = toWorld.Forward();
		pRec.pdf = m_invSurfaceArea;
		pRec.measure = EArea;
		return Spectrum(1.0f);
	}

	CUDA_FUNC_IN Spectrum evalPosition(const PositionSamplingRecord &pRec) const
	{
		return Spectrum((pRec.measure == EArea) ? m_invSurfaceArea : 0.0f);
	}

	CUDA_FUNC_IN float pdfPosition(const PositionSamplingRecord &pRec) const
	{
		return (pRec.measure == EArea) ? m_invSurfaceArea : 0.0f;
	}

	CUDA_FUNC_IN Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const
	{
		dRec.d = pRec.n;
		dRec.measure = EDiscrete;
		dRec.pdf = 1.0f;

		return Spectrum(1.0f);
	}

	CUDA_FUNC_IN float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		return (pRec.measure == EDiscrete) ? 1.0f : 0.0f;
	}

	CUDA_FUNC_IN Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		return Spectrum((pRec.measure == EDiscrete) ? 1.0f : 0.0f);
	}

	CUDA_FUNC_IN bool getSamplePosition(const PositionSamplingRecord &pRec, const DirectionSamplingRecord &dRec, float2 &samplePosition) const
	{
		float3 localP = toWorld.Inverse() * pRec.p;
		float3 sample = m_cameraToSample * localP;

		if (sample.x < 0 || sample.x > 1 || sample.y < 0 || sample.y > 1)
			return false;

		samplePosition = make_float2(sample.x * m_resolution.x,
		                        sample.y * m_resolution.y);
		return true;
	}

	float4x4 getProjectionMatrix() const
	{
		return float4x4::Orthographic(screenScale.x, screenScale.y, m_fNearFarDepths.x, m_fNearFarDepths.y);
	}

	TYPE_FUNC(e_OrthographicCamera)
};

#define e_TelecentricCamera_TYPE 5
struct e_TelecentricCamera : public e_SensorBase
{
protected:
	virtual void updateData();
	float4x4 m_cameraToSample;
	float4x4 m_sampleToCamera;
	float m_normalization;
	float2 screenScale;
	float m_aperturePdf;
public:
	e_TelecentricCamera(int w, int h, float a, float dist, float sx = 1, float sy = 1)
		: e_SensorBase(false, true)
	{
		m_fNearFarDepths = make_float2(0.00001f, 100000);
		SetFilmData(w, h);
		SetScreenScale(sx, sy);
		m_apertureRadius = a;
		m_focusDistance = dist;
	}

	void SetScreenScale(float sx, float sy)
	{
		screenScale = make_float2(sx, sy);
		updateData();
	}

	CUDA_FUNC_IN Spectrum sampleRay(Ray &ray, const float2 &pixelSample, const float2 &apertureSample) const
	{
		float2 diskSample = Warp::squareToUniformDiskConcentric(apertureSample)
			* (m_apertureRadius / screenScale.x);

		/* Compute the corresponding position on the
		   near plane (in local camera space) */
		float3 focusP = m_sampleToCamera * make_float3(
			pixelSample.x * m_invResolution.x,
			pixelSample.y * m_invResolution.y, 0.0f);
		focusP.z = m_focusDistance;

		/* Compute the ray origin */
		float3 orig = make_float3(diskSample.x+focusP.x,
			diskSample.y+focusP.y, 0.0f);

		ray = Ray(toWorld * orig, toWorld.TransformNormal(focusP - orig));

		return Spectrum(1.0f);
	}

	CUDA_FUNC_IN Spectrum eval(const Ray& r, const TraceResult& r2, const float3 &d, float2 &samplePos) const
	{
		return Spectrum(0.0f);
	}

	CUDA_FUNC_IN Spectrum samplePosition(PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const
	{
		float a = sample.x + 1.0f, b = sample.y + 1.0f;
		unsigned int tmp1 = *(unsigned int*)&a & 0x7FFFFF;
		unsigned int tmp2 = *(unsigned int*)&b & 0x7FFFFF;

		float rand1 = (tmp1 >> 11)   * (1.0f / 0xFFF);
		float rand2 = (tmp2 >> 11)   * (1.0f / 0xFFF);
		float rand3 = (tmp1 & 0x7FF) * (1.0f / 0x7FF);
		float rand4 = (tmp2 & 0x7FF) * (1.0f / 0x7FF);

		float2 aperturePos = Warp::squareToUniformDiskConcentric(make_float2(rand1, rand2))
			* (m_apertureRadius / screenScale.x);
		float2 samplePos = make_float2(rand3, rand4);

		if (extra) {
			/* The caller wants to condition on a specific pixel position */
			pRec.uv = *extra + samplePos;
			samplePos.x = pRec.uv.x * m_invResolution.x;
			samplePos.y = pRec.uv.y * m_invResolution.y;
		}

		float3 p = m_sampleToCamera * make_float3(
			aperturePos.x + samplePos.x, aperturePos.y + samplePos.y, 0.0f);

		pRec.p = toWorld * make_float3(p.x, p.y, 0.0f);
		pRec.n = toWorld.Forward();
		pRec.pdf = m_aperturePdf;
		pRec.measure = EArea;
		return Spectrum(1.0f);
	}

	CUDA_FUNC_IN Spectrum evalPosition(const PositionSamplingRecord &pRec) const
	{
		return Spectrum((pRec.measure == EArea) ? m_aperturePdf : 0.0f);
	}

	CUDA_FUNC_IN float pdfPosition(const PositionSamplingRecord &pRec) const
	{
		return (pRec.measure == EArea) ? m_aperturePdf : 0.0f;
	}

	CUDA_FUNC_IN Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const
	{
		float3 nearP = m_sampleToCamera * make_float3(sample.x, sample.y, 0.0f);

		/* Turn that into a normalized ray direction */
		float3 d = normalize(nearP);
		dRec.d = toWorld.TransformNormal(d);
		dRec.measure = ESolidAngle;
		dRec.pdf = m_normalization / (d.z * d.z * d.z);

		return Spectrum(1.0f);
	}

	CUDA_FUNC_IN float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		if (dRec.measure != ESolidAngle)
			return 0.0f;
		return 1.0f;
	}

	CUDA_FUNC_IN Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		if (dRec.measure != ESolidAngle)
			return Spectrum(0.0f);
		return Spectrum(1.0f);
	}

	CUDA_FUNC_IN bool getSamplePosition(const PositionSamplingRecord &pRec, const DirectionSamplingRecord &dRec, float2 &samplePosition) const
	{
		return false;
	}

	float4x4 getProjectionMatrix() const
	{
		return float4x4::Orthographic(screenScale.x, screenScale.y, m_fNearFarDepths.x, m_fNearFarDepths.y);
	}

	TYPE_FUNC(e_TelecentricCamera)
};

#define CAM_SIZE DMAX5(sizeof(e_SphericalCamera), sizeof(e_PerspectiveCamera), sizeof(e_ThinLensCamera), sizeof(e_OrthographicCamera), sizeof(e_TelecentricCamera))

struct e_Sensor
{
private:
	CUDA_ALIGN(16) unsigned char Data[CAM_SIZE];
	unsigned int type;
public:
	//storage for the last viewing frustum, might(!) be computed, don't depend on it
	AABB m_sLastFrustum;

	float4x4 View() const;

	float3 Position() const;

	void SetToWorld(const float3& pos, const float4x4& rot);

	void SetToWorld(const float3& pos, const float3& f);

	void SetFilmData(int w, int h);

	void SetToWorld(const float4x4& w);

	float4x4 getGLViewProjection() const;

	float4x4 getProjectionMatrix() const
	{
		CALL_FUNC5(e_SphericalCamera,e_PerspectiveCamera,e_ThinLensCamera,e_OrthographicCamera,e_TelecentricCamera, getProjectionMatrix())
		return float4x4::Identity();
	}

	CUDA_FUNC_IN Ray GenRay(int x, int y)
	{
		Ray r;
		sampleRay(r,make_float2(x,y),make_float2(0,0));
		return r;
	}

	CUDA_FUNC_IN Spectrum sampleRay(Ray &ray, const float2 &pixelSample, const float2 &apertureSample) const
	{
		CALL_FUNC5(e_SphericalCamera,e_PerspectiveCamera,e_ThinLensCamera,e_OrthographicCamera,e_TelecentricCamera, sampleRay(ray, pixelSample, apertureSample))
		return Spectrum(1.0f);
	}

	CUDA_FUNC_IN Spectrum eval(const Ray& r, const TraceResult& r2, const float3 &d, float2 &samplePos) const
	{
		CALL_FUNC5(e_SphericalCamera,e_PerspectiveCamera,e_ThinLensCamera,e_OrthographicCamera,e_TelecentricCamera, eval(r, r2, d, samplePos))
		return Spectrum(1.0f);
	}

	CUDA_FUNC_IN Spectrum samplePosition(PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const
	{
		CALL_FUNC5(e_SphericalCamera,e_PerspectiveCamera,e_ThinLensCamera,e_OrthographicCamera,e_TelecentricCamera, samplePosition(pRec, sample, extra))
		return Spectrum(1.0f);
	}

	CUDA_FUNC_IN Spectrum evalPosition(const PositionSamplingRecord &pRec) const
	{
		CALL_FUNC5(e_SphericalCamera,e_PerspectiveCamera,e_ThinLensCamera,e_OrthographicCamera,e_TelecentricCamera, evalPosition(pRec))
		return Spectrum(1.0f);
	}

	CUDA_FUNC_IN float pdfPosition(const PositionSamplingRecord &pRec) const
	{
		CALL_FUNC5(e_SphericalCamera,e_PerspectiveCamera,e_ThinLensCamera,e_OrthographicCamera,e_TelecentricCamera, pdfPosition(pRec))
		return 1.0f;
	}

	CUDA_FUNC_IN Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const
	{
		CALL_FUNC5(e_SphericalCamera,e_PerspectiveCamera,e_ThinLensCamera,e_OrthographicCamera,e_TelecentricCamera, sampleDirection(dRec, pRec, sample, extra))
		return Spectrum(1.0f);
	}

	CUDA_FUNC_IN float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		CALL_FUNC5(e_SphericalCamera,e_PerspectiveCamera,e_ThinLensCamera,e_OrthographicCamera,e_TelecentricCamera, pdfDirection(dRec, pRec))
		return 1.0f;
	}

	CUDA_FUNC_IN Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		CALL_FUNC5(e_SphericalCamera,e_PerspectiveCamera,e_ThinLensCamera,e_OrthographicCamera,e_TelecentricCamera, evalDirection(dRec, pRec))
		return Spectrum(1.0f);
	}

	CUDA_FUNC_IN bool getSamplePosition(const PositionSamplingRecord &pRec, const DirectionSamplingRecord &dRec, float2 &samplePosition) const
	{
		CALL_FUNC5(e_SphericalCamera,e_PerspectiveCamera,e_ThinLensCamera,e_OrthographicCamera,e_TelecentricCamera, getSamplePosition(pRec, dRec, samplePosition))
		return false;
	}

	STD_VIRTUAL_SET_BASE(e_SensorBase)
};