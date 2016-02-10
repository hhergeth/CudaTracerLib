#pragma once

#include <Math/float4x4.h>
#include <Math/Ray.h>
#include <Math/Spectrum.h>
#include <Math/AABB.h>
#include "Samples.h"
#include "AbstractEmitter.h"
#include <VirtualFuncType.h>

namespace CudaTracerLib {

struct Frame;

enum ESensorFlags
{
	/// Sensor response contains a Dirac delta term with respect to time
	EDeltaTime = 0x010,

	/// Does the \ref sampleRay() function need an aperture sample?
	ENeedsApertureSample = 0x020,

	/// Is the sensor a projective camera?
	EProjectiveCamera = 0x100,

	/// Is the sensor a perspective camera?
	EPerspectiveSensor = 0x200,

	/// Is the sensor an orthographic camera?
	EOrthographicSensor = 0x400,

	/// Does the sample given to \ref samplePosition() determine the pixel coordinates
	EPositionSampleMapsToPixels = 0x1000,

	/// Does the sample given to \ref sampleDirection() determine the pixel coordinates
	EDirectionSampleMapsToPixels = 0x2000
};

struct SensorBase : public AbstractEmitter//, public BaseTypeHelper<5459539>
{
public:
	float aspect;
	Vec2f m_resolution, m_invResolution;
	NormalizedT<OrthogonalAffineMap> toWorld;
	float4x4 m_cameraToSample, m_sampleToCamera;
	Vec2f m_fNearFarDepths;
	float fov;
	float m_apertureRadius;
	float m_focusDistance;
public:
	SensorBase()
		: AbstractEmitter(0)
	{

	}
	SensorBase(unsigned int type)
		: AbstractEmitter(type)
	{
		toWorld = NormalizedT<OrthogonalAffineMap>::Identity();
	}
	virtual void Update()
	{
		m_invResolution = Vec2f(1) / m_resolution;
		aspect = m_resolution.x / m_resolution.y;
	}
	virtual void SetToWorld(const NormalizedT<OrthogonalAffineMap>& w)
	{
		toWorld = w;
		Update();
	}
	///_fov in degrees
	virtual void SetFov(float _fov)
	{
		fov = math::Radians(_fov);
		Update();
	}
	virtual void SetNearFarDepth(float nearD, float farD)
	{
		m_fNearFarDepths = Vec2f(nearD, farD);
		Update();
	}
	virtual void SetFilmData(int w, int h)
	{
		m_resolution = Vec2f((float)w, (float)h);
		m_invResolution = Vec2f(1) / m_resolution;
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
	NormalizedT<OrthogonalAffineMap> getWorld()
	{
		return toWorld;
	}
};

struct SphericalSensor : public SensorBase//, public e_DerivedTypeHelper<1>
{
	TYPE_FUNC(1)
	SphericalSensor()
		: SensorBase(EDeltaPosition | EDirectionSampleMapsToPixels)
	{
		
	}

	SphericalSensor(int w, int h)
		: SensorBase(EDeltaPosition | EDirectionSampleMapsToPixels)
	{
		SetFilmData(w, h);
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleRay(NormalizedT<Ray> &ray, const Vec2f &pixelSample, const Vec2f &apertureSample) const;

	CUDA_FUNC_IN Spectrum sampleRayDifferential(NormalizedT<Ray> &ray, NormalizedT<Ray> &rayX, NormalizedT<Ray> &rayY, const Vec2f &pixelSample, const Vec2f &apertureSample) const
	{
		return sampleRay(ray, pixelSample, apertureSample);
	}

	CUDA_FUNC_IN Spectrum eval(const Vec3f& p, const Frame& sys, const NormalizedT<Vec3f> &d) const
	{
		return Spectrum(0.0f);
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const;

	CUDA_FUNC_IN float pdfDirect(const DirectSamplingRecord &dRec) const
	{
		return (dRec.measure == EDiscrete) ? 1.0f : 0.0f;
	}

	CUDA_FUNC_IN Spectrum samplePosition(PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const
	{
		pRec.p = toWorld.Translation();
		pRec.n = NormalizedT<Vec3f>(0.0f);
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

	CUDA_FUNC_IN Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const
	{
		Vec3f samplePos = Vec3f(sample.x, sample.y, 0.0f);

		if (extra)
		{
			/* The caller wants to condition on a specific pixel position */
			samplePos.x = (extra->x + sample.x) * m_invResolution.x;
			samplePos.y = (extra->y + sample.y) * m_invResolution.y;
		}

		pRec.uv = Vec2f(samplePos.x * m_resolution.x, samplePos.y * m_resolution.y);

		float sinPhi, cosPhi, sinTheta, cosTheta;
		sincos(samplePos.x * 2 * PI, &sinPhi, &cosPhi);
		sincos(samplePos.y * PI, &sinTheta, &cosTheta);

		dRec.d = toWorld.TransformPoint(Vec3f(sinPhi*sinTheta, cosTheta, -cosPhi*sinTheta)).normalized();
		dRec.measure = ESolidAngle;
		dRec.pdf = 1 / (2 * PI * PI * max(sinTheta, EPSILON));

		return Spectrum(1.0f);
	}

	CUDA_DEVICE CUDA_HOST float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const;

	CUDA_DEVICE CUDA_HOST Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const;

	CUDA_DEVICE CUDA_HOST bool getSamplePosition(const PositionSamplingRecord &pRec, const DirectionSamplingRecord &dRec, Vec2f &samplePosition) const;
};

struct PerspectiveSensor : public SensorBase//, public e_DerivedTypeHelper<2>
{
	TYPE_FUNC(2)
	Vec3f m_dx, m_dy;
	float m_normalization;
	AABB m_imageRect;
public:
	PerspectiveSensor()
		: SensorBase(EDeltaPosition | EPerspectiveSensor | EOnSurface | EDirectionSampleMapsToPixels | EProjectiveCamera)
	{
		
	}
	///_fov in degrees
	PerspectiveSensor(int w, int h, float _fov)
		: SensorBase(EDeltaPosition | EPerspectiveSensor | EOnSurface | EDirectionSampleMapsToPixels | EProjectiveCamera)
	{
		m_fNearFarDepths = Vec2f(1, 100000);
		SetFilmData(w, h);
		fov = math::Radians(_fov);
	}

	virtual void Update();

	CUDA_DEVICE CUDA_HOST float importance(const NormalizedT<Vec3f> &d) const;

	CUDA_DEVICE CUDA_HOST Spectrum sampleRay(NormalizedT<Ray> &ray, const Vec2f &pixelSample, const Vec2f &apertureSample) const;

	CUDA_DEVICE CUDA_HOST Spectrum sampleRayDifferential(NormalizedT<Ray> &ray, NormalizedT<Ray> &rayX, NormalizedT<Ray> &rayY, const Vec2f &pixelSample, const Vec2f &apertureSample) const;

	CUDA_FUNC_IN Spectrum eval(const Vec3f& p, const Frame& sys, const NormalizedT<Vec3f> &d) const
	{
		return Spectrum(0.0f);
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const;

	CUDA_FUNC_IN float pdfDirect(const DirectSamplingRecord &dRec) const
	{
		return (dRec.measure == EDiscrete) ? 1.0f : 0.0f;
	}

	CUDA_FUNC_IN Spectrum samplePosition(PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const
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

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const;

	CUDA_FUNC_IN float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		if (dRec.measure != ESolidAngle)
			return 0.0f;
		
		return importance(toWorld.TransformDirectionTranspose(dRec.d));
	}

	CUDA_FUNC_IN Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		if (dRec.measure != ESolidAngle)
			return Spectrum(0.0f);

		return importance(toWorld.TransformDirectionTranspose(dRec.d));
	}

	CUDA_DEVICE CUDA_HOST bool getSamplePosition(const PositionSamplingRecord &pRec, const DirectionSamplingRecord &dRec, Vec2f &samplePosition) const;
};

struct ThinLensSensor : public SensorBase//, public e_DerivedTypeHelper<3>
{
	TYPE_FUNC(3)
	Vec3f m_dx, m_dy;
	float m_aperturePdf;
	float m_normalization;
	CUDA_DEVICE CUDA_HOST float importance(const Vec3f &p, const NormalizedT<Vec3f> &d, Vec2f* sample = 0) const;
public:
	ThinLensSensor()
		: SensorBase(ENeedsApertureSample | EPerspectiveSensor | EOnSurface | EDirectionSampleMapsToPixels | EProjectiveCamera)
	{
		
	}
	///_fov in degrees
	ThinLensSensor(int w, int h, float _fov, float a, float dist)
		: SensorBase(ENeedsApertureSample | EPerspectiveSensor | EOnSurface | EDirectionSampleMapsToPixels | EProjectiveCamera)
	{
		m_fNearFarDepths = Vec2f(1, 100000);
		SetFilmData(w, h);
		fov = _fov;
		m_apertureRadius = a;
		m_focusDistance = dist;
	}

	virtual void Update();

	CUDA_DEVICE CUDA_HOST Spectrum sampleRay(NormalizedT<Ray> &ray, const Vec2f &pixelSample, const Vec2f &apertureSample) const;

	CUDA_DEVICE CUDA_HOST Spectrum sampleRayDifferential(NormalizedT<Ray> &ray, NormalizedT<Ray> &rayX, NormalizedT<Ray> &rayY, const Vec2f &pixelSample, const Vec2f &apertureSample) const;

	CUDA_FUNC_IN Spectrum eval(const Vec3f& p, const Frame& sys, const NormalizedT<Vec3f> &d) const
	{
		return Spectrum(0.0f);
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const;

	CUDA_DEVICE CUDA_HOST float pdfDirect(const DirectSamplingRecord &dRec) const;

	CUDA_DEVICE CUDA_HOST Spectrum samplePosition(PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const;

	CUDA_FUNC_IN Spectrum evalPosition(const PositionSamplingRecord &pRec) const
	{
		return Spectrum((pRec.measure == EArea) ? m_aperturePdf : 0.0f);
	}

	CUDA_FUNC_IN float pdfPosition(const PositionSamplingRecord &pRec) const
	{
		return (pRec.measure == EArea) ? m_aperturePdf : 0.0f;
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const;

	CUDA_FUNC_IN float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		if (dRec.measure != ESolidAngle)
			return 0.0f;

		return importance(toWorld.TransformPointTranspose(pRec.p), toWorld.TransformDirectionTranspose(dRec.d));
	}

	CUDA_FUNC_IN Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		if (dRec.measure != ESolidAngle)
			return Spectrum(0.0f);

		return importance(toWorld.TransformPointTranspose(pRec.p), toWorld.TransformDirectionTranspose(dRec.d));
	}

	CUDA_FUNC_IN bool getSamplePosition(const PositionSamplingRecord &pRec, const DirectionSamplingRecord &dRec, Vec2f &samplePosition) const
	{
		auto localP = toWorld.TransformPointTranspose(pRec.p);
		auto localD = toWorld.TransformDirectionTranspose(dRec.d);

		if (localD.z <= 0)
			return false;

		Vec3f intersection = localP + localD * (m_focusDistance / localD.z);

		Vec3f screenSample = m_cameraToSample.TransformPoint(intersection);
		if (screenSample.x < 0 || screenSample.x > 1 ||
			screenSample.y < 0 || screenSample.y > 1)
			return false;

		samplePosition = Vec2f(
			screenSample.x * m_resolution.x,
			screenSample.y * m_resolution.y);

		return true;
	}
};

struct OrthographicSensor : public SensorBase//, public e_DerivedTypeHelper<4>
{
	TYPE_FUNC(4)
public:
	Vec2f screenScale;
private:
	float m_invSurfaceArea, m_scale;
	Vec3f m_dx, m_dy;
public:
	OrthographicSensor()
		: SensorBase(EDeltaDirection | EOrthographicSensor | EPositionSampleMapsToPixels | EProjectiveCamera)
	{
		
	}

	OrthographicSensor(int w, int h, float sx = 1, float sy = 1)
		: SensorBase(EDeltaDirection | EOrthographicSensor | EPositionSampleMapsToPixels | EProjectiveCamera)
	{
		m_fNearFarDepths = Vec2f(0.00001f, 100000);
		SetFilmData(w, h);
		SetScreenScale(sx, sy);
	}

	void SetScreenScale(float sx, float sy)
	{
		screenScale = Vec2f(sx, sy);
		Update();
	}

	virtual void Update();

	CUDA_DEVICE CUDA_HOST Spectrum sampleRay(NormalizedT<Ray> &ray, const Vec2f &pixelSample, const Vec2f &apertureSample) const;

	CUDA_DEVICE CUDA_HOST Spectrum sampleRayDifferential(NormalizedT<Ray> &ray, NormalizedT<Ray> &rayX, NormalizedT<Ray> &rayY, const Vec2f &pixelSample, const Vec2f &apertureSample) const;

	CUDA_FUNC_IN Spectrum eval(const Vec3f& p, const Frame& sys, const NormalizedT<Vec3f> &d) const
	{
		return Spectrum(0.0f);
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirect(DirectSamplingRecord &dRec, const Vec2f &) const;

	CUDA_FUNC_IN float pdfDirect(const DirectSamplingRecord &dRec) const
	{
		return (dRec.measure == EDiscrete) ? 1.0f : 0.0f;
	}

	CUDA_DEVICE CUDA_HOST Spectrum samplePosition(PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const;

	CUDA_FUNC_IN Spectrum evalPosition(const PositionSamplingRecord &pRec) const
	{
		return Spectrum((pRec.measure == EArea) ? m_invSurfaceArea : 0.0f);
	}

	CUDA_FUNC_IN float pdfPosition(const PositionSamplingRecord &pRec) const
	{
		return (pRec.measure == EArea) ? m_invSurfaceArea : 0.0f;
	}

	CUDA_FUNC_IN Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const
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

	CUDA_DEVICE CUDA_HOST bool getSamplePosition(const PositionSamplingRecord &pRec, const DirectionSamplingRecord &dRec, Vec2f &samplePosition) const;
};

struct TelecentricSensor : public SensorBase//, public e_DerivedTypeHelper<5>
{
	TYPE_FUNC(5)
public:
	Vec2f screenScale;
protected:
	float m_normalization;
	float m_aperturePdf;
	Vec3f m_dx, m_dy;
public:
	TelecentricSensor()
		: SensorBase(ENeedsApertureSample | EOrthographicSensor | EPositionSampleMapsToPixels | EProjectiveCamera)
	{
		
	}

	TelecentricSensor(int w, int h, float a, float dist, float sx = 1, float sy = 1)
		: SensorBase(ENeedsApertureSample | EOrthographicSensor | EPositionSampleMapsToPixels | EProjectiveCamera)
	{
		m_fNearFarDepths = Vec2f(0.00001f, 100000);
		SetFilmData(w, h);
		SetScreenScale(sx, sy);
		m_apertureRadius = a;
		m_focusDistance = dist;
	}

	void SetScreenScale(float sx, float sy)
	{
		screenScale = Vec2f(sx, sy);
		Update();
	}

	virtual void Update();

	CUDA_DEVICE CUDA_HOST Spectrum sampleRay(NormalizedT<Ray> &ray, const Vec2f &pixelSample, const Vec2f &apertureSample) const;

	CUDA_DEVICE CUDA_HOST Spectrum sampleRayDifferential(NormalizedT<Ray> &ray, NormalizedT<Ray> &rayX, NormalizedT<Ray> &rayY, const Vec2f &pixelSample, const Vec2f &apertureSample) const;

	CUDA_FUNC_IN Spectrum eval(const Vec3f& p, const Frame& sys, const NormalizedT<Vec3f> &d) const
	{
		return Spectrum(0.0f);
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const;

	CUDA_FUNC_IN float pdfDirect(const DirectSamplingRecord &dRec) const
	{
		return 0.0f;
	}

	CUDA_DEVICE CUDA_HOST Spectrum samplePosition(PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const;

	CUDA_FUNC_IN Spectrum evalPosition(const PositionSamplingRecord &pRec) const
	{
		return Spectrum((pRec.measure == EArea) ? m_aperturePdf : 0.0f);
	}

	CUDA_FUNC_IN float pdfPosition(const PositionSamplingRecord &pRec) const
	{
		return (pRec.measure == EArea) ? m_aperturePdf : 0.0f;
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const;

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

	CUDA_FUNC_IN bool getSamplePosition(const PositionSamplingRecord &pRec, const DirectionSamplingRecord &dRec, Vec2f &samplePosition) const
	{
		return false;
	}
};

struct Sensor : public CudaVirtualAggregate<SensorBase, SphericalSensor, PerspectiveSensor, ThinLensSensor, OrthographicSensor, TelecentricSensor>
{
public:
	//storage for the last viewing frustum, might(!) be computed, don't depend on it
	NormalizedT<OrthogonalAffineMap> View() const;

	Vec3f Position() const;

	void SetToWorld(const Vec3f& pos, const NormalizedT<OrthogonalAffineMap>& rot);

	void SetToWorld(const Vec3f& pos, const Vec3f& f);

	void SetToWorld(const Vec3f& pos, const Vec3f& tar, const Vec3f& up);

	void SetFilmData(int w, int h);

	void SetToWorld(const NormalizedT<OrthogonalAffineMap>& w);

	float4x4 getProjectionMatrix() const;

	CUDA_FUNC_IN NormalizedT<Ray> GenRay(int x, int y)
	{
		NormalizedT<Ray> r;
		sampleRay(r, Vec2f((float)x, (float)y), Vec2f(0, 0));
		return r;
	}

	CALLER(sampleRay)
	CUDA_FUNC_IN Spectrum sampleRay(NormalizedT<Ray> &ray, const Vec2f &pixelSample, const Vec2f &apertureSample) const
	{
		return sampleRay_Helper::Caller<Spectrum>(this, ray, pixelSample, apertureSample);
	}

	CALLER(sampleRayDifferential)
	CUDA_FUNC_IN Spectrum sampleRayDifferential(NormalizedT<Ray> &ray, NormalizedT<Ray> &rayX, NormalizedT<Ray> &rayY, const Vec2f &pixelSample, const Vec2f &apertureSample) const
	{
		return sampleRayDifferential_Helper::Caller<Spectrum>(this, ray, rayX, rayY, pixelSample, apertureSample);
	}

	CALLER(eval)
	CUDA_FUNC_IN Spectrum eval(const Vec3f& p, const Frame& sys, const NormalizedT<Vec3f> &d) const
	{
		return eval_Helper::Caller<Spectrum>(this, p, sys, d);
	}

	CALLER(sampleDirect)
	CUDA_FUNC_IN Spectrum sampleDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const
	{
		return sampleDirect_Helper::Caller<Spectrum>(this, dRec, sample);
	}

	CALLER(pdfDirect)
	CUDA_FUNC_IN float pdfDirect(const DirectSamplingRecord &dRec) const
	{
		return pdfDirect_Helper::Caller<float>(this, dRec);
	}

	CALLER(samplePosition)
	CUDA_FUNC_IN Spectrum samplePosition(PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const
	{
		return samplePosition_Helper::Caller<Spectrum>(this, pRec, sample, extra);
	}

	CALLER(evalPosition)
	CUDA_FUNC_IN Spectrum evalPosition(const PositionSamplingRecord &pRec) const
	{
		return evalPosition_Helper::Caller<Spectrum>(this, pRec);
	}

	CALLER(pdfPosition)
	CUDA_FUNC_IN float pdfPosition(const PositionSamplingRecord &pRec) const
	{
		return pdfPosition_Helper::Caller<float>(this, pRec);
	}

	CALLER(sampleDirection)
	CUDA_FUNC_IN Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const
	{
		return sampleDirection_Helper::Caller<Spectrum>(this, dRec, pRec, sample, extra);
	}

	CALLER(pdfDirection)
	CUDA_FUNC_IN float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		return pdfDirection_Helper::Caller<float>(this, dRec, pRec);
	}

	CALLER(evalDirection)
	CUDA_FUNC_IN Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		return evalDirection_Helper::Caller<Spectrum>(this, dRec, pRec);
	}

	CALLER(getSamplePosition)
	CUDA_FUNC_IN bool getSamplePosition(const PositionSamplingRecord &pRec, const DirectionSamplingRecord &dRec, Vec2f &samplePosition) const
	{
		return getSamplePosition_Helper::Caller<bool>(this, pRec, dRec, samplePosition);
	}
};

}