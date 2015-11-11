#include "e_Sensor.h"

namespace CudaTracerLib {

Spectrum e_SphericalCamera::sampleRay(Ray &ray, const Vec2f &pixelSample, const Vec2f &apertureSample) const
{
	float sinPhi, cosPhi, sinTheta, cosTheta;
	sincos((1.0f - pixelSample.x * m_invResolution.x) * 2 * PI, &sinPhi, &cosPhi);
	sincos((1.0f - pixelSample.y * m_invResolution.y) * PI, &sinTheta, &cosTheta);

	Vec3f d = Vec3f(sinPhi*sinTheta, cosTheta, -cosPhi*sinTheta);
	ray = Ray(toWorld.Translation(), toWorld.TransformDirection(d));

	return Spectrum(1.0f);
}

Spectrum e_SphericalCamera::sampleDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const
{
	Vec3f refP = toWorldInverse.TransformPoint(dRec.ref);
	Vec3f d(refP);
	float dist = length(d), invDist = 1.0f / dist;
	d *= invDist;

	dRec.uv = Vec2f(
		math::modulo(atan2f(d.x, -d.z) * INV_TWOPI, 1.0f) * m_resolution.x,
		(1.0f - math::safe_acos(d.y) * INV_PI) * m_resolution.y
		);

	float sinTheta = math::safe_sqrt(1 - d.y*d.y);

	dRec.p = toWorld.Translation();
	dRec.d = (dRec.p - dRec.ref) * invDist;
	dRec.dist = dist;
	dRec.n = Vec3f(0.0f);
	dRec.pdf = 1;
	dRec.measure = EDiscrete;

	return Spectrum((1 / (2 * PI * PI * max(sinTheta, EPSILON))) * invDist * invDist);
}

float e_SphericalCamera::pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
{
	if (dRec.measure != ESolidAngle)
		return 0.0f;

	Vec3f d = toWorldInverse.TransformDirection(dRec.d);
	float sinTheta = math::safe_sqrt(1 - d.y*d.y);

	return 1 / (2 * PI * PI * max(sinTheta, EPSILON));
}

Spectrum e_SphericalCamera::evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
{
	if (dRec.measure != ESolidAngle)
		return Spectrum(0.0f);

	Vec3f d = toWorldInverse.TransformDirection(dRec.d);
	float sinTheta = math::safe_sqrt(1 - d.y*d.y);

	return Spectrum(1 / (2 * PI * PI * max(sinTheta, EPSILON)));
}

bool e_SphericalCamera::getSamplePosition(const PositionSamplingRecord &pRec, const DirectionSamplingRecord &dRec, Vec2f &samplePosition) const
{
	Vec3f d = normalize(toWorldInverse.TransformDirection(dRec.d));

	samplePosition = Vec2f(
		math::modulo(atan2(d.x, -d.z) * INV_TWOPI, (float)1) * m_resolution.x,
		(1.0f - math::safe_acos(d.y) * INV_PI) * m_resolution.y
		);

	return true;
}

void e_PerspectiveCamera::Update()
{
	e_SensorBase::Update();
	m_cameraToSample =
		float4x4::Scale(Vec3f(-0.5f, -0.5f*aspect, 1.0f))
		% float4x4::Translate(Vec3f(-1.0f, -1.0f / aspect, 0.0f))
		% float4x4::Perspective(fov, m_fNearFarDepths.x, m_fNearFarDepths.y);

	m_sampleToCamera = m_cameraToSample.inverse();

	m_dx = m_sampleToCamera.TransformPoint(Vec3f(m_invResolution.x, 0.0f, 0.0f))
		- m_sampleToCamera.TransformPoint(Vec3f(0.0f));
	m_dy = m_sampleToCamera.TransformPoint(Vec3f(0.0f, m_invResolution.y, 0.0f))
		- m_sampleToCamera.TransformPoint(Vec3f(0.0f));

	Vec3f	min = m_sampleToCamera.TransformPoint(Vec3f(0, 0, 0)),
		max = m_sampleToCamera.TransformPoint(Vec3f(1, 1, 0));
	m_imageRect = AABB(min / min.z, max / max.z);
	m_imageRect.minV.z = -FLT_MAX; m_imageRect.maxV.z = FLT_MAX;
	m_normalization = 1.0f / (m_imageRect.Size().x * m_imageRect.Size().y);
}

float e_PerspectiveCamera::importance(const Vec3f &d) const
{
	float cosTheta = Frame::cosTheta(d);

	/* Check if the direction points behind the camera */
	if (cosTheta <= 0)
		return 0.0f;

	/* Compute the position on the plane at distance 1 */
	float invCosTheta = 1.0f / cosTheta;
	Vec2f p = Vec2f(d.x * invCosTheta, d.y * invCosTheta);

	/* Check if the point lies inside the chosen crop rectangle */
	if (!m_imageRect.Contains(Vec3f(p, 0)))
		return 0.0f;
	return invCosTheta * invCosTheta * invCosTheta * m_normalization;
}

Spectrum e_PerspectiveCamera::sampleRay(Ray &ray, const Vec2f &pixelSample, const Vec2f &apertureSample) const
{
	Vec3f nearP = m_sampleToCamera.TransformPoint(Vec3f(
		pixelSample.x * m_invResolution.x,
		pixelSample.y * m_invResolution.y, 0.0f));

	/* Turn that into a normalized ray direction, and
		adjust the ray interval accordingly */
	Vec3f d = normalize(nearP);
	ray = Ray(toWorld.Translation(), toWorld.TransformDirection(d));

	return Spectrum(1.0f);
}

Spectrum e_PerspectiveCamera::sampleRayDifferential(Ray &ray, Ray &rayX, Ray &rayY, const Vec2f &pixelSample, const Vec2f &apertureSample) const
{
	Vec3f nearP = m_sampleToCamera.TransformPoint(Vec3f(
		pixelSample.x * m_invResolution.x,
		pixelSample.y * m_invResolution.y, 0.0f));

	Vec3f d = normalize(nearP);
	ray = Ray(toWorld.Translation(), toWorld.TransformDirection(d));

	rayX.origin = rayY.origin = ray.origin;

	rayX.direction = toWorld.TransformDirection(normalize(nearP + m_dx));
	rayY.direction = toWorld.TransformDirection(normalize(nearP + m_dy));
	return Spectrum(1.0f);
}

Spectrum e_PerspectiveCamera::sampleDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const
{
	Vec3f refP = toWorldInverse.TransformPoint(dRec.ref);

	/* Check if it is outside of the clip range */
	if (refP.z < m_fNearFarDepths.x || refP.z > m_fNearFarDepths.y) {
		dRec.pdf = 0.0f;
		return Spectrum(0.0f);
	}

	Vec3f screenSample = m_cameraToSample.TransformPoint(refP);
	dRec.uv = Vec2f(screenSample.x, screenSample.y);
	if (dRec.uv.x < 0 || dRec.uv.x  > 1 ||
		dRec.uv.y < 0 || dRec.uv.y > 1) {
		dRec.pdf = 0.0f;
		return Spectrum(0.0f);
	}

	dRec.uv.x *= m_resolution.x;
	dRec.uv.y *= m_resolution.y;

	Vec3f localD = refP;
	float dist = length(localD),
		invDist = 1.0f / dist;
	localD *= invDist;

	dRec.p = toWorld.Translation();
	dRec.d = invDist * (dRec.p - dRec.ref);
	dRec.dist = dist;
	dRec.n = toWorld.Forward();
	dRec.pdf = 1;
	dRec.measure = EDiscrete;

	return Spectrum(importance(localD)*invDist*invDist);
}

Spectrum e_PerspectiveCamera::sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const
{
	Vec3f samplePos = Vec3f(sample.x, sample.y, 0.0f);

	if (extra) {
		/* The caller wants to condition on a specific pixel position */
		samplePos.x = (extra->x + sample.x) * m_invResolution.x;
		samplePos.y = (extra->y + sample.y) * m_invResolution.y;
	}

	pRec.uv = Vec2f(samplePos.x * m_resolution.x,
		samplePos.y * m_resolution.y);

	/* Compute the corresponding position on the
		near plane (in local camera space) */
	Vec3f nearP = m_sampleToCamera.TransformPoint(samplePos);

	/* Turn that into a normalized ray direction */
	Vec3f d = normalize(nearP);
	dRec.d = toWorld.TransformDirection(d);
	dRec.measure = ESolidAngle;
	dRec.pdf = m_normalization / (d.z * d.z * d.z);

	return Spectrum(1.0f);
}

bool e_PerspectiveCamera::getSamplePosition(const PositionSamplingRecord &pRec, const DirectionSamplingRecord &dRec, Vec2f &samplePosition) const
{
	Vec3f local = toWorldInverse.TransformDirection(dRec.d);

	if (local.z <= 0)
		return false;

	Vec3f screenSample = m_cameraToSample.TransformPoint(local);
	if (screenSample.x < 0 || screenSample.x > 1 ||
		screenSample.y < 0 || screenSample.y > 1)
		return false;

	samplePosition = Vec2f(
		screenSample.x * m_resolution.x,
		screenSample.y * m_resolution.y);

	return true;
}

void e_ThinLensCamera::Update()
{
	e_SensorBase::Update();
	m_cameraToSample =
		float4x4::Scale(Vec3f(-0.5f, -0.5f*aspect, 1.0f))
		% float4x4::Translate(Vec3f(-1.0f, -1.0f / aspect, 0.0f))
		% float4x4::Perspective(fov, m_fNearFarDepths.x, m_fNearFarDepths.y);
	m_sampleToCamera = m_cameraToSample.inverse();

	m_dx = m_sampleToCamera.TransformPoint(Vec3f(m_invResolution.x, 0.0f, 0.0f))
		- m_sampleToCamera.TransformPoint(Vec3f(0.0f));
	m_dy = m_sampleToCamera.TransformPoint(Vec3f(0.0f, m_invResolution.y, 0.0f))
		- m_sampleToCamera.TransformPoint(Vec3f(0.0f));

	m_aperturePdf = 1 / (PI * m_apertureRadius * m_apertureRadius);

	Vec3f	min = m_sampleToCamera.TransformPoint(Vec3f(0, 0, 0)),
		max = m_sampleToCamera.TransformPoint(Vec3f(1, 1, 0));
	AABB m_imageRect = AABB(min / min.z, max / max.z);
	m_normalization = 1.0f / (m_imageRect.Size().x * m_imageRect.Size().y);
}

float e_ThinLensCamera::importance(const Vec3f &p, const Vec3f &d, Vec2f* sample) const
{
	float cosTheta = Frame::cosTheta(d);
	if (cosTheta <= 0)
		return 0.0f;
	float invCosTheta = 1.0f / cosTheta;
	Vec3f scr = m_cameraToSample.TransformPoint(p + d * (m_focusDistance*invCosTheta));
	if (scr.x < 0 || scr.x > 1 ||
		scr.y < 0 || scr.y > 1)
		return 0.0f;

	if (sample) {
		sample->x = scr.x * m_resolution.x;
		sample->y = scr.y * m_resolution.y;
	}

	return m_normalization * invCosTheta * invCosTheta * invCosTheta;
}

Spectrum e_ThinLensCamera::sampleRay(Ray &ray, const Vec2f &pixelSample, const Vec2f &apertureSample) const
{
	Vec2f tmp = Warp::squareToUniformDiskConcentric(apertureSample) * m_apertureRadius;

	/* Compute the corresponding position on the
		near plane (in local camera space) */
	Vec3f nearP = m_sampleToCamera.TransformPoint(Vec3f(
		pixelSample.x * m_invResolution.x,
		pixelSample.y * m_invResolution.y, 0.0f));

	/* Aperture position */
	Vec3f apertureP = Vec3f(tmp.x, tmp.y, 0.0f);

	/* Sampled position on the focal plane */
	Vec3f focusP = nearP * (m_focusDistance / nearP.z);

	/* Turn these into a normalized ray direction, and
		adjust the ray interval accordingly */
	Vec3f d = normalize(focusP - apertureP);

	ray = Ray(toWorld.TransformPoint(apertureP), toWorld.TransformDirection(d));

	return Spectrum(1.0f);
}

Spectrum e_ThinLensCamera::sampleRayDifferential(Ray &ray, Ray &rayX, Ray &rayY, const Vec2f &pixelSample, const Vec2f &apertureSample) const
{
	Vec2f tmp = Warp::squareToUniformDiskConcentric(apertureSample) * m_apertureRadius;
	Vec3f nearP = m_sampleToCamera.TransformPoint(Vec3f(
		pixelSample.x * m_invResolution.x,
		pixelSample.y * m_invResolution.y, 0.0f));
	Vec3f apertureP = Vec3f(tmp.x, tmp.y, 0.0f);

	float fDist = m_focusDistance / nearP.z;
	Vec3f focusP = nearP       * fDist;
	Vec3f focusPx = (nearP + m_dx) * fDist;
	Vec3f focusPy = (nearP + m_dy) * fDist;

	Vec3f d = normalize(focusP - apertureP);
	ray = Ray(toWorld.TransformPoint(apertureP), toWorld.TransformDirection(d));
	rayX.origin = rayY.origin = ray.origin;
	rayX.direction = toWorld.TransformDirection(normalize(focusPx - apertureP));
	rayY.direction = toWorld.TransformDirection(normalize(focusPy - apertureP));
	return Spectrum(1.0f);
}

Spectrum e_ThinLensCamera::sampleDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const
{
	Vec3f refP = toWorldInverse.TransformPoint(dRec.ref);

	/* Check if it is outside of the clip range */
	if (refP.z < m_fNearFarDepths.x || refP.z > m_fNearFarDepths.y) {
		dRec.pdf = 0.0f;
		return Spectrum(0.0f);
	}

	/* Sample a position on the aperture (in local coordinates) */
	Vec2f tmp = Warp::squareToUniformDiskConcentric(sample) * m_apertureRadius;
	Vec3f apertureP = Vec3f(tmp.x, tmp.y, 0);

	/* Compute the normalized direction vector from the
		aperture position to the reference point */
	Vec3f localD = (refP - apertureP);
	float dist = length(localD),
		invDist = 1.0f / dist;
	localD *= invDist;

	float value = importance(apertureP, localD, &dRec.uv);
	if (value == 0.0f) {
		dRec.pdf = 0.0f;
		return Spectrum(0.0f);
	}

	dRec.p = toWorld.TransformPoint(apertureP);
	dRec.d = (dRec.p - dRec.ref) * invDist;
	dRec.dist = dist;
	dRec.n = toWorld.Forward();
	dRec.pdf = m_aperturePdf * dist*dist / (Frame::cosTheta(localD));
	dRec.measure = ESolidAngle;

	/* intentionally missing a cosine factor wrt. the aperture
		disk (it is already accounted for in importance()) */
	return Spectrum(value * invDist * invDist);
}

float e_ThinLensCamera::pdfDirect(const DirectSamplingRecord &dRec) const
{
	float dp = -dot(dRec.n, dRec.d);
	if (dp < 0)
		return 0.0f;

	if (dRec.measure == ESolidAngle)
		return m_aperturePdf * dRec.dist*dRec.dist / dp;
	else if (dRec.measure == EArea)
		return m_aperturePdf;
	else
		return 0.0f;
}

Spectrum e_ThinLensCamera::sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const
{
	Vec3f samplePos = Vec3f(sample.x, sample.y, 0.0f);

	if (extra) {
		/* The caller wants to condition on a specific pixel position */
		samplePos.x = (extra->x + sample.x) * m_invResolution.x;
		samplePos.y = (extra->y + sample.y) * m_invResolution.y;
	}

	pRec.uv = Vec2f(samplePos.x * m_resolution.x,
		samplePos.y * m_resolution.y);

	/* Compute the corresponding position on the
		near plane (in local camera space) */
	Vec3f nearP = m_sampleToCamera.TransformPoint(samplePos);
	nearP.x = nearP.x * (m_focusDistance / nearP.z);
	nearP.y = nearP.y * (m_focusDistance / nearP.z);
	nearP.z = m_focusDistance;

	Vec3f apertureP = toWorldInverse.TransformPoint(pRec.p);

	/* Turn that into a normalized ray direction */
	Vec3f d = normalize(nearP - apertureP);
	dRec.d = toWorld.TransformDirection(d);
	dRec.measure = ESolidAngle;
	dRec.pdf = m_normalization / (d.z * d.z * d.z);

	return Spectrum(1.0f);
}

void e_OrthographicCamera::Update()
{
	e_SensorBase::Update();
	m_cameraToSample =
		float4x4::Scale(Vec3f(-0.5f, -0.5f*aspect, 1.0f))
		% float4x4::Translate(Vec3f(-1.0f, -1.0f / aspect, 0.0f))
		% float4x4::orthographic(m_fNearFarDepths.x, m_fNearFarDepths.y);

	m_sampleToCamera = m_cameraToSample.inverse();

	m_dx = m_sampleToCamera.TransformPoint(Vec3f(m_invResolution.x, 0.0f, 0.0f))
		- m_sampleToCamera.TransformPoint(Vec3f(0.0f));
	m_dy = m_sampleToCamera.TransformPoint(Vec3f(0.0f, m_invResolution.y, 0.0f))
		- m_sampleToCamera.TransformPoint(Vec3f(0.0f));

	m_invSurfaceArea = 1.0f / (
		length(toWorld.TransformPoint(m_sampleToCamera.Right())) *
		length(toWorld.TransformPoint(m_sampleToCamera.Up())));
	m_scale = length(toWorld.Forward());
}

Spectrum e_OrthographicCamera::sampleRay(Ray &ray, const Vec2f &pixelSample, const Vec2f &apertureSample) const
{
	Vec3f nearP = m_sampleToCamera.TransformPoint(Vec3f(
		pixelSample.x * m_invResolution.x,
		pixelSample.y * m_invResolution.y, 0.0f));

	ray = Ray(toWorld.TransformPoint(Vec3f(nearP.x, nearP.y, 0.0f)), toWorld.Forward());

	return Spectrum(1.0f);
}

Spectrum e_OrthographicCamera::sampleRayDifferential(Ray &ray, Ray &rayX, Ray &rayY, const Vec2f &pixelSample, const Vec2f &apertureSample) const
{
	Vec3f nearP = m_sampleToCamera.TransformPoint(Vec3f(
		pixelSample.x * m_invResolution.x,
		pixelSample.y * m_invResolution.y, 0.0f));
	ray = Ray(toWorld.TransformPoint(nearP), toWorld.Forward());
	rayX.origin = toWorld.TransformPoint(nearP + m_dx);
	rayY.origin = toWorld.TransformPoint(nearP + m_dy);
	rayX.direction = rayY.direction = ray.direction;
	return Spectrum(1.0f);
}

Spectrum e_OrthographicCamera::sampleDirect(DirectSamplingRecord &dRec, const Vec2f &) const
{
	dRec.n = toWorld.Forward();
	float scale = length(dRec.n);

	Vec3f localP = toWorldInverse.TransformPoint(dRec.ref);
	localP.z *= scale;

	Vec3f sample = m_cameraToSample.TransformPoint(localP);

	if (sample.x < 0 || sample.x > 1 || sample.y < 0 ||
		sample.y > 1 || sample.z < 0 || sample.z > 1) {
		dRec.pdf = 0.0f;
		return Spectrum(0.0f);
	}

	dRec.p = toWorld.TransformPoint(Vec3f(localP.x, localP.y, 0.0f));
	dRec.n /= scale;
	dRec.d = -dRec.n;
	dRec.dist = localP.z;
	dRec.uv = Vec2f(sample.x * m_resolution.x,
		sample.y * m_resolution.y);
	dRec.pdf = 1.0f;
	dRec.measure = EDiscrete;

	return Spectrum(m_invSurfaceArea);
}

Spectrum e_OrthographicCamera::samplePosition(PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const
{
	Vec3f samplePos = Vec3f(sample.x, sample.y, 0.0f);

	if (extra) {
		/* The caller wants to condition on a specific pixel position */
		samplePos.x = (extra->x + sample.x) * m_invResolution.x;
		samplePos.y = (extra->y + sample.y) * m_invResolution.y;
	}

	pRec.uv = Vec2f(samplePos.x * m_resolution.x, samplePos.y * m_resolution.y);

	Vec3f nearP = m_sampleToCamera.TransformPoint(samplePos);

	nearP.z = 0.0f;
	pRec.p = toWorld.TransformPoint(nearP);
	pRec.n = toWorld.Forward();
	pRec.pdf = m_invSurfaceArea;
	pRec.measure = EArea;
	return Spectrum(1.0f);
}

bool e_OrthographicCamera::getSamplePosition(const PositionSamplingRecord &pRec, const DirectionSamplingRecord &dRec, Vec2f &samplePosition) const
{
	Vec3f localP = toWorldInverse.TransformPoint(pRec.p);
	Vec3f sample = m_cameraToSample.TransformPoint(localP);

	if (sample.x < 0 || sample.x > 1 || sample.y < 0 || sample.y > 1)
		return false;

	samplePosition = Vec2f(sample.x * m_resolution.x,
		sample.y * m_resolution.y);
	return true;
}

void e_TelecentricCamera::Update()
{
	e_SensorBase::Update();
	m_cameraToSample =
		float4x4::Scale(Vec3f(-0.5f, -0.5f*aspect, 1.0f))
		% float4x4::Translate(Vec3f(-1.0f, -1.0f / aspect, 0.0f))
		% float4x4::orthographic(m_fNearFarDepths.x, m_fNearFarDepths.y);

	m_sampleToCamera = m_cameraToSample.inverse();

	m_dx = m_sampleToCamera.TransformPoint(Vec3f(m_invResolution.x, 0.0f, 0.0f))
		- m_sampleToCamera.TransformPoint(Vec3f(0.0f));
	m_dy = m_sampleToCamera.TransformPoint(Vec3f(0.0f, m_invResolution.y, 0.0f))
		- m_sampleToCamera.TransformPoint(Vec3f(0.0f));

	m_normalization = 1.0f / (
		length(toWorld.TransformPoint(m_sampleToCamera.Right())) *
		length(toWorld.TransformPoint(m_sampleToCamera.Up())));

	m_aperturePdf = 1.0f / (PI * m_apertureRadius * m_apertureRadius);
}

Spectrum e_TelecentricCamera::sampleRay(Ray &ray, const Vec2f &pixelSample, const Vec2f &apertureSample) const
{
	Vec2f diskSample = Warp::squareToUniformDiskConcentric(apertureSample)
		* (m_apertureRadius / screenScale.x);

	/* Compute the corresponding position on the
		near plane (in local camera space) */
	Vec3f focusP = m_sampleToCamera.TransformPoint(Vec3f(
		pixelSample.x * m_invResolution.x,
		pixelSample.y * m_invResolution.y, 0.0f));
	focusP.z = m_focusDistance;

	/* Compute the ray origin */
	Vec3f orig = Vec3f(diskSample.x + focusP.x,
		diskSample.y + focusP.y, 0.0f);

	ray = Ray(toWorld.TransformPoint(orig), toWorld.TransformDirection(focusP - orig));

	return Spectrum(1.0f);
}

Spectrum e_TelecentricCamera::sampleRayDifferential(Ray &ray, Ray &rayX, Ray &rayY, const Vec2f &pixelSample, const Vec2f &apertureSample) const
{
	Vec2f diskSample = Warp::squareToUniformDiskConcentric(apertureSample) * (m_apertureRadius / screenScale.x);
	Vec3f focusP = m_sampleToCamera.TransformPoint(Vec3f(
		pixelSample.x * m_invResolution.x,
		pixelSample.y * m_invResolution.y, 0.0f));
	focusP.z = m_focusDistance;
	/* Compute the ray origin */
	Vec3f orig = Vec3f(diskSample.x + focusP.x,
		diskSample.y + focusP.y, 0.0f);
	ray = Ray(toWorld.TransformPoint(orig), toWorld.TransformDirection(focusP - orig));
	rayX.origin = toWorld.TransformPoint(orig + m_dx);
	rayY.origin = toWorld.TransformPoint(orig + m_dy);
	rayX.direction = rayY.direction = ray.direction;
	return Spectrum(1.0f);
}

Spectrum e_TelecentricCamera::sampleDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const
{
	float f = m_focusDistance, apertureRadius = m_apertureRadius / screenScale.x;

	Vec3f localP = toWorldInverse.TransformPoint(dRec.ref);

	float dist = localP.z;
	if (dist < m_fNearFarDepths.x || dist > m_fNearFarDepths.y) {
		dRec.pdf = 0.0f;
		return Spectrum(0.0f);
	}

	/* Circle of confusion */
	float radius = math::abs(localP.z - f) * apertureRadius / f;
	radius += apertureRadius;

	/* Sample the ray origin */
	Vec2f disk = Warp::squareToUniformDiskConcentric(sample);
	Vec3f diskP = Vec3f(disk.x*radius + localP.x, disk.y*radius + localP.y, 0.0f);

	/* Compute the intersection with the focal plane */
	Vec3f localD = localP - diskP;
	Vec3f intersection = diskP + localD * (f / localD.z);

	/* Determine the associated sample coordinates */
	Vec3f uv = m_cameraToSample.TransformPoint(intersection);
	if (uv.x < 0 || uv.x > 1 || uv.y < 0 || uv.y > 1) {
		dRec.pdf = 0.0f;
		return Spectrum(0.0f);
	}

	dRec.uv = Vec2f(uv.x, uv.y);
	dRec.p = toWorld.TransformPoint(diskP);
	dRec.n = toWorld.Forward();
	dRec.d = dRec.p - dRec.ref;
	dRec.dist = length(dRec.d);
	dRec.d /= dRec.dist;
	dRec.measure = ESolidAngle;

	dRec.pdf = dist*dist / (-dot(dRec.n, dRec.d)* PI * radius*radius);

	return Spectrum(m_normalization);
}

Spectrum e_TelecentricCamera::samplePosition(PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const
{
	float a = sample.x + 1.0f, b = sample.y + 1.0f;
	unsigned int tmp1 = *(unsigned int*)&a & 0x7FFFFF;
	unsigned int tmp2 = *(unsigned int*)&b & 0x7FFFFF;

	float rand1 = (tmp1 >> 11)   * (1.0f / 0xFFF);
	float rand2 = (tmp2 >> 11)   * (1.0f / 0xFFF);
	float rand3 = (tmp1 & 0x7FF) * (1.0f / 0x7FF);
	float rand4 = (tmp2 & 0x7FF) * (1.0f / 0x7FF);

	Vec2f aperturePos = Warp::squareToUniformDiskConcentric(Vec2f(rand1, rand2))
		* (m_apertureRadius / screenScale.x);
	Vec2f samplePos = Vec2f(rand3, rand4);

	if (extra) {
		/* The caller wants to condition on a specific pixel position */
		pRec.uv = *extra + samplePos;
		samplePos.x = pRec.uv.x * m_invResolution.x;
		samplePos.y = pRec.uv.y * m_invResolution.y;
	}

	Vec3f p = m_sampleToCamera.TransformPoint(Vec3f(
		aperturePos.x + samplePos.x, aperturePos.y + samplePos.y, 0.0f));

	pRec.p = toWorld.TransformPoint(Vec3f(p.x, p.y, 0.0f));
	pRec.n = toWorld.Forward();
	pRec.pdf = m_aperturePdf;
	pRec.measure = EArea;
	return Spectrum(1.0f);
}

Spectrum e_TelecentricCamera::sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const
{
	Vec3f nearP = m_sampleToCamera.TransformPoint(Vec3f(sample.x, sample.y, 0.0f));

	/* Turn that into a normalized ray direction */
	Vec3f d = normalize(nearP);
	dRec.d = toWorld.TransformDirection(d);
	dRec.measure = ESolidAngle;
	dRec.pdf = m_normalization / (d.z * d.z * d.z);

	return Spectrum(1.0f);
}

float4x4 e_Sensor::View() const
{
	return As<e_SensorBase>()->getWorld();
}

Vec3f e_Sensor::Position() const
{
	return As<e_SensorBase>()->getWorld().Translation();
}

void e_Sensor::SetToWorld(const Vec3f& pos, const float4x4& _rot)
{
	float4x4 rot = _rot;
	rot.col(3, Vec4f(0, 0, 0, 1));
	rot.row(3, Vec4f(0, 0, 0, 1));
	SetToWorld(float4x4::Translate(pos) % rot);
}

void e_Sensor::SetToWorld(const Vec3f& pos, const Vec3f& _f)
{
	Vec3f f = normalize(_f);
	Vec3f r = normalize(cross(f, Vec3f(0, 1, 0)));
	Vec3f u = normalize(cross(r, f));
	SetToWorld(pos, pos + f, u);
}

void e_Sensor::SetToWorld(const Vec3f& pos, const Vec3f& tar, const Vec3f& u)
{
	Vec3f f = normalize(tar - pos);
	Vec3f r = normalize(cross(f, u));
	float4x4 m_mView = float4x4::Identity();
	m_mView.col(0, Vec4f(r, 0));
	m_mView.col(1, Vec4f(u, 0));
	m_mView.col(2, Vec4f(f, 0));
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

float4x4 e_Sensor::getProjectionMatrix() const
{
	float4x4 q = float4x4::Translate(-1, -1, 0) % float4x4::Scale(Vec3f(2, 2, 1)) % As()->m_cameraToSample;
	return q;
}

}