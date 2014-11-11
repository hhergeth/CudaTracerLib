#pragma once

#include "..\MathTypes.h"
#include "e_Samples.h"

//this architecture and the implementations are completly copied from mitsuba!

struct e_SensorBase : public e_BaseType
{
public:
	float aspect;
	float2 m_resolution, m_invResolution;
	float4x4 toWorld, toWorldInverse;
	float4x4 m_cameraToSample, m_sampleToCamera;
	float2 m_fNearFarDepths;
	float fov;
	float m_apertureRadius;
	float m_focusDistance;
public:
	const bool isFieldOfViewSensor;
	const bool isAppertureSensor;
	e_SensorBase()
		: isFieldOfViewSensor(false), isAppertureSensor(true)
	{

	}
	e_SensorBase(bool fov, bool app)
		: isFieldOfViewSensor(fov), isAppertureSensor(app)
	{
		toWorld = float4x4::Identity();
	}
	virtual void Update()
	{
		toWorldInverse = toWorld.inverse();
		m_invResolution = make_float2(1) / m_resolution;
		aspect = m_resolution.x / m_resolution.y;
	}
	virtual void SetToWorld(const float4x4& w)
	{
		toWorld = w;
		toWorldInverse = w.inverse();
		Update();
	}
	///_fov in degrees
	virtual void SetFov(float _fov)
	{
		fov = Radians(_fov);
		Update();
	}
	virtual void SetNearFarDepth(float nearD, float farD)
	{
		m_fNearFarDepths = make_float2(nearD, farD);
		Update();
	}
	virtual void SetFilmData(int w, int h)
	{
		m_resolution = make_float2(w, h);
		m_invResolution = make_float2(1) / m_resolution;
		aspect = m_resolution.x / m_resolution.y;
		Update();
	}
	virtual void SetApperture(float a)
	{
		m_apertureRadius = a;
		Update();
	}
	virtual void SetFocalDistance(float a)
	{
		m_focusDistance = a;
		Update();
	}
	float4x4 getWorld()
	{
		return toWorld;
	}
};

#define e_SphericalCamera_TYPE 1
struct e_SphericalCamera : public e_SensorBase
{
	e_SphericalCamera()
		: e_SensorBase(false, false)
	{}
	e_SphericalCamera(int w, int h)
		: e_SensorBase(false, false)
	{
		SetFilmData(w, h);
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleRay(Ray &ray, const float2 &pixelSample, const float2 &apertureSample) const;

	CUDA_FUNC_IN Spectrum sampleRayDifferential(Ray &ray, Ray &rayX, Ray &rayY, const float2 &pixelSample, const float2 &apertureSample) const
	{
		return sampleRay(ray, pixelSample, apertureSample);
	}

	CUDA_FUNC_IN Spectrum eval(const float3& p, const Frame& sys, const float3 &d) const
	{
		return Spectrum(0.0f);
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirect(DirectSamplingRecord &dRec, const float2 &sample) const;

	CUDA_FUNC_IN float pdfDirect(const DirectSamplingRecord &dRec) const
	{
		return (dRec.measure == EDiscrete) ? 1.0f : 0.0f;
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

		dRec.d = toWorld.TransformPoint(make_float3(sinPhi*sinTheta, cosTheta, -cosPhi*sinTheta));
		dRec.measure = ESolidAngle;
		dRec.pdf = 1 / (2 * PI * PI * MAX(sinTheta, EPSILON));

		return Spectrum(1.0f);
	}

	CUDA_DEVICE CUDA_HOST float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const;

	CUDA_DEVICE CUDA_HOST Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const;

	CUDA_DEVICE CUDA_HOST bool getSamplePosition(const PositionSamplingRecord &pRec, const DirectionSamplingRecord &dRec, float2 &samplePosition) const;

	TYPE_FUNC(e_SphericalCamera)
};

#define e_PerspectiveCamera_TYPE 2
struct e_PerspectiveCamera : public e_SensorBase
{
	float3 m_dx, m_dy;
	float m_normalization;
	AABB m_imageRect;
public:
	e_PerspectiveCamera()
		: e_SensorBase(true, false)
	{}
	///_fov in degrees
	e_PerspectiveCamera(int w, int h, float _fov)
		: e_SensorBase(true, false)
	{
		m_fNearFarDepths = make_float2(1, 100000);
		SetFilmData(w, h);
		fov = Radians(_fov);
	}

	virtual void Update();

	CUDA_DEVICE CUDA_HOST float importance(const float3 &d) const;

	CUDA_DEVICE CUDA_HOST Spectrum sampleRay(Ray &ray, const float2 &pixelSample, const float2 &apertureSample) const;

	CUDA_DEVICE CUDA_HOST Spectrum sampleRayDifferential(Ray &ray, Ray &rayX, Ray &rayY, const float2 &pixelSample, const float2 &apertureSample) const;

	CUDA_FUNC_IN Spectrum eval(const float3& p, const Frame& sys, const float3 &d) const
	{
		return Spectrum(0.0f);
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirect(DirectSamplingRecord &dRec, const float2 &sample) const;

	CUDA_FUNC_IN float pdfDirect(const DirectSamplingRecord &dRec) const
	{
		return (dRec.measure == EDiscrete) ? 1.0f : 0.0f;
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

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const;

	CUDA_FUNC_IN float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		if (dRec.measure != ESolidAngle)
			return 0.0f;

		return importance(toWorldInverse.TransformDirection(dRec.d));
	}

	CUDA_FUNC_IN Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		if (dRec.measure != ESolidAngle)
			return Spectrum(0.0f);

		return Spectrum(importance(toWorldInverse.TransformDirection(dRec.d)));
	}

	CUDA_DEVICE CUDA_HOST bool getSamplePosition(const PositionSamplingRecord &pRec, const DirectionSamplingRecord &dRec, float2 &samplePosition) const;

	TYPE_FUNC(e_PerspectiveCamera)
};

#define e_ThinLensCamera_TYPE 3
struct e_ThinLensCamera : public e_SensorBase
{
	float3 m_dx, m_dy;
	float m_aperturePdf;
	float m_normalization;
	CUDA_DEVICE CUDA_HOST float importance(const float3 &p, const float3 &d, float2* sample = 0) const;
public:
	e_ThinLensCamera()
		: e_SensorBase(true, true)
	{}
	///_fov in degrees
	e_ThinLensCamera(int w, int h, float _fov, float a, float dist)
		: e_SensorBase(true, true)
	{
		m_fNearFarDepths = make_float2(1, 100000);
		SetFilmData(w, h);
		fov = _fov;
		m_apertureRadius = a;
		m_focusDistance = dist;
	}

	virtual void Update();

	CUDA_DEVICE CUDA_HOST Spectrum sampleRay(Ray &ray, const float2 &pixelSample, const float2 &apertureSample) const;

	CUDA_DEVICE CUDA_HOST Spectrum sampleRayDifferential(Ray &ray, Ray &rayX, Ray &rayY, const float2 &pixelSample, const float2 &apertureSample) const;

	CUDA_FUNC_IN Spectrum eval(const float3& p, const Frame& sys, const float3 &d) const
	{
		return Spectrum(0.0f);
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirect(DirectSamplingRecord &dRec, const float2 &sample) const;

	CUDA_DEVICE CUDA_HOST float pdfDirect(const DirectSamplingRecord &dRec) const;

	CUDA_FUNC_IN Spectrum samplePosition(PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const
	{
		float2 aperturePos = Warp::squareToUniformDiskConcentric(sample) * m_apertureRadius;

		pRec.p = toWorld.TransformPoint(make_float3(aperturePos.x, aperturePos.y, 0.0f));
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

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const;

	CUDA_FUNC_IN float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		if (dRec.measure != ESolidAngle)
			return 0.0f;

		return importance(toWorldInverse.TransformPoint(pRec.p), toWorldInverse.TransformDirection(dRec.d));
	}

	CUDA_FUNC_IN Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		if (dRec.measure != ESolidAngle)
			return Spectrum(0.0f);

		return Spectrum(importance(toWorldInverse.TransformPoint(pRec.p), toWorldInverse.TransformDirection(dRec.d)));
	}

	CUDA_FUNC_IN bool getSamplePosition(const PositionSamplingRecord &pRec, const DirectionSamplingRecord &dRec, float2 &samplePosition) const
	{
		float3 localP(toWorldInverse.TransformPoint(pRec.p));
		float3 localD(toWorldInverse.TransformDirection(dRec.d));

		if (localD.z <= 0)
			return false;

		float3 intersection = localP + localD * (m_focusDistance / localD.z);

		float3 screenSample = m_cameraToSample.TransformPoint(intersection);
		if (screenSample.x < 0 || screenSample.x > 1 ||
			screenSample.y < 0 || screenSample.y > 1)
			return false;

		samplePosition = make_float2(
				screenSample.x * m_resolution.x,
				screenSample.y * m_resolution.y);

		return true;
	}

	TYPE_FUNC(e_ThinLensCamera)
};

#define e_OrthographicCamera_TYPE 4
struct e_OrthographicCamera : public e_SensorBase
{
public:
	float2 screenScale;
private:
	float m_invSurfaceArea, m_scale;
	float3 m_dx, m_dy;
public:
	e_OrthographicCamera()
		: e_SensorBase(false, false)
	{}
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
		Update();
	}

	virtual void Update();

	CUDA_DEVICE CUDA_HOST Spectrum sampleRay(Ray &ray, const float2 &pixelSample, const float2 &apertureSample) const;

	CUDA_DEVICE CUDA_HOST Spectrum sampleRayDifferential(Ray &ray, Ray &rayX, Ray &rayY, const float2 &pixelSample, const float2 &apertureSample) const;

	CUDA_FUNC_IN Spectrum eval(const float3& p, const Frame& sys, const float3 &d) const
	{
		return Spectrum(0.0f);
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirect(DirectSamplingRecord &dRec, const float2 &) const;

	CUDA_FUNC_IN float pdfDirect(const DirectSamplingRecord &dRec) const
	{
		return (dRec.measure == EDiscrete) ? 1.0f : 0.0f;
	}

	CUDA_DEVICE CUDA_HOST Spectrum samplePosition(PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const;

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

	CUDA_DEVICE CUDA_HOST bool getSamplePosition(const PositionSamplingRecord &pRec, const DirectionSamplingRecord &dRec, float2 &samplePosition) const;

	TYPE_FUNC(e_OrthographicCamera)
};

#define e_TelecentricCamera_TYPE 5
struct e_TelecentricCamera : public e_SensorBase
{
public:
	float2 screenScale;
protected:
	float m_normalization;
	float m_aperturePdf;
	float3 m_dx, m_dy;
public:
	e_TelecentricCamera()
		: e_SensorBase(false, true)
	{}
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
		Update();
	}

	virtual void Update();

	CUDA_DEVICE CUDA_HOST Spectrum sampleRay(Ray &ray, const float2 &pixelSample, const float2 &apertureSample) const;

	CUDA_DEVICE CUDA_HOST Spectrum sampleRayDifferential(Ray &ray, Ray &rayX, Ray &rayY, const float2 &pixelSample, const float2 &apertureSample) const;

	CUDA_FUNC_IN Spectrum eval(const float3& p, const Frame& sys, const float3 &d) const
	{
		return Spectrum(0.0f);
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirect(DirectSamplingRecord &dRec, const float2 &sample) const;

	CUDA_FUNC_IN float pdfDirect(const DirectSamplingRecord &dRec) const
	{
		return 0.0f;
	}

	CUDA_DEVICE CUDA_HOST Spectrum samplePosition(PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const;

	CUDA_FUNC_IN Spectrum evalPosition(const PositionSamplingRecord &pRec) const
	{
		return Spectrum((pRec.measure == EArea) ? m_aperturePdf : 0.0f);
	}

	CUDA_FUNC_IN float pdfPosition(const PositionSamplingRecord &pRec) const
	{
		return (pRec.measure == EArea) ? m_aperturePdf : 0.0f;
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const;

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

	TYPE_FUNC(e_TelecentricCamera)
};

#define CAM_SIZE DMAX5(sizeof(e_SphericalCamera), sizeof(e_PerspectiveCamera), sizeof(e_ThinLensCamera), sizeof(e_OrthographicCamera), sizeof(e_TelecentricCamera))

struct e_Sensor : public e_AggregateBaseType<e_SensorBase, CAM_SIZE>
{
public:
	//storage for the last viewing frustum, might(!) be computed, don't depend on it
	AABB m_sLastFrustum;

	float4x4 View() const;

	float3 Position() const;

	void SetToWorld(const float3& pos, const float4x4& rot);

	void SetToWorld(const float3& pos, const float3& f);

	void SetToWorld(const float3& pos, const float3& tar, const float3& up);

	void SetFilmData(int w, int h);

	void SetToWorld(const float4x4& w);

	float4x4 getProjectionMatrix() const;

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

	CUDA_FUNC_IN Spectrum sampleRayDifferential(Ray &ray, Ray &rayX, Ray &rayY, const float2 &pixelSample, const float2 &apertureSample) const
	{
		CALL_FUNC5(e_SphericalCamera, e_PerspectiveCamera, e_ThinLensCamera, e_OrthographicCamera, e_TelecentricCamera, sampleRayDifferential(ray, rayX, rayY, pixelSample, apertureSample))
			return Spectrum(1.0f);
	}

	CUDA_FUNC_IN Spectrum eval(const float3& p, const Frame& sys, const float3 &d) const
	{
		CALL_FUNC5(e_SphericalCamera,e_PerspectiveCamera,e_ThinLensCamera,e_OrthographicCamera,e_TelecentricCamera, eval(p, sys, d))
		return Spectrum(1.0f);
	}

	CUDA_FUNC_IN Spectrum sampleDirect(DirectSamplingRecord &dRec, const float2 &sample) const
	{
		CALL_FUNC5(e_SphericalCamera,e_PerspectiveCamera,e_ThinLensCamera,e_OrthographicCamera,e_TelecentricCamera, sampleDirect(dRec, sample))
		return Spectrum(1.0f);
	}

	CUDA_FUNC_IN float pdfDirect(const DirectSamplingRecord &dRec) const
	{
		CALL_FUNC5(e_SphericalCamera,e_PerspectiveCamera,e_ThinLensCamera,e_OrthographicCamera,e_TelecentricCamera, pdfDirect(dRec))
		return 0.0f;
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
};