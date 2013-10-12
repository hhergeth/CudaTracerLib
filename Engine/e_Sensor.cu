#include "e_Sensor.h"

Spectrum e_SphericalCamera::sampleRay(Ray &ray, const float2 &pixelSample, const float2 &apertureSample) const
{
	float sinPhi, cosPhi, sinTheta, cosTheta;
	sincos((1.0f - pixelSample.x * m_invResolution.x) * 2 * PI, &sinPhi, &cosPhi);
	sincos((1.0f - pixelSample.y * m_invResolution.y) * PI, &sinTheta, &cosTheta);

	float3 d = make_float3(sinPhi*sinTheta, cosTheta, -cosPhi*sinTheta);
	ray = Ray(toWorld.Translation(), toWorld.TransformNormal(d));

	return Spectrum(1.0f);
}

Spectrum e_SphericalCamera::sampleDirect(DirectSamplingRecord &dRec, const float2 &sample) const
{
	float3 refP = toWorldInverse * dRec.ref;
	float3 d(refP);
	float dist = length(d), invDist = 1.0f / dist;
	d *= invDist;

	dRec.uv = make_float2(
		math::modulo(atan2f(d.x, -d.z) * INV_TWOPI, 1.0f) * m_resolution.x,
		math::safe_acos(d.y) * INV_PI * m_resolution.y
	);

	float sinTheta = math::safe_sqrt(1-d.y*d.y);

	dRec.p = toWorld.Translation();
	dRec.d = (dRec.p - dRec.ref) * invDist;
	dRec.dist = dist;
	dRec.n = make_float3(0.0f);
	dRec.pdf = 1;
	dRec.measure = EDiscrete;

	return Spectrum((1/(2 * PI * PI * MAX(sinTheta, EPSILON))) * invDist * invDist);
}

float e_SphericalCamera::pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
{
	if (dRec.measure != ESolidAngle)
		return 0.0f;

	float3 d = toWorldInverse.TransformNormal(dRec.d);
	float sinTheta = math::safe_sqrt(1-d.y*d.y);

	return 1 / (2 * PI * PI * MAX(sinTheta, EPSILON));
}

Spectrum e_SphericalCamera::evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
{
	if (dRec.measure != ESolidAngle)
		return Spectrum(0.0f);

	float3 d = toWorldInverse.TransformNormal(dRec.d);
	float sinTheta = math::safe_sqrt(1-d.y*d.y);

	return Spectrum(1 / (2 * PI * PI * MAX(sinTheta, EPSILON)));
}

bool e_SphericalCamera::getSamplePosition(const PositionSamplingRecord &pRec, const DirectionSamplingRecord &dRec, float2 &samplePosition) const
{
	float3 d = normalize(toWorldInverse.TransformNormal(dRec.d));

	samplePosition = make_float2(
		math::modulo(atan2(d.x, -d.z) * INV_TWOPI, (float) 1) * m_resolution.x,
		math::safe_acos(d.y) * INV_PI * m_resolution.y
	);

	return true;
}

float4x4 persp(float fov, float clipNear, float clipFar)
{
	float recip = 1.0f / (clipFar - clipNear);

	/* Perform a scale so that the field of view is mapped
	 * to the interval [-1, 1] */
	float cot = 1.0f / tanf(fov/2.0f);

	float4x4 trafo(
		cot,  0,    0,   0,
		0,    cot,  0,   0,
		0,    0,    clipFar * recip, -clipNear * clipFar * recip,
		0,    0,    1,   0
	);
	return trafo.Transpose();
}
void e_PerspectiveCamera::Update()
{
	e_SensorBase::Update();
	float2 relSize = make_float2(1), relOffset = make_float2(0);
	m_cameraToSample =
			  persp(fov, m_fNearFarDepths.x, m_fNearFarDepths.y)
			* float4x4::Translate(make_float3(1.0f, 1.0f/aspect, 0.0f))
			* float4x4::Scale(make_float3(0.5f, 0.5f*aspect, 1.0f))
			* float4x4::Translate(make_float3(-relOffset.x, -relOffset.y, 0.0f))
			* float4x4::Scale(make_float3(1.0f / relSize.x, 1.0f / relSize.y, 1.0f));
	m_sampleToCamera = m_cameraToSample.Inverse();

	m_dx = m_sampleToCamera * make_float3(m_invResolution.x, 0.0f, 0.0f)
			- m_sampleToCamera * make_float3(0.0f);
	m_dy = m_sampleToCamera * make_float3(0.0f, m_invResolution.y, 0.0f)
			- m_sampleToCamera * make_float3(0.0f);

	float3 min = m_sampleToCamera * make_float3(0, 0, 0),
		   max = m_sampleToCamera * make_float3(1, 1, 0);
	m_imageRect = AABB(min / min.z, max / max.z);
	m_imageRect.minV.z = -FLT_MAX; m_imageRect.maxV.z = FLT_MAX;
	m_normalization = 1.0f / (m_imageRect.Size().x * m_imageRect.Size().y);

//	DirectSamplingRecord dRec(make_float3(301.48853f,398.27206f,559.20007f),make_float3(0),make_float2(0));
	//sampleDirect(dRec, make_float2(0));
}

float e_PerspectiveCamera::importance(const float3 &d) const
{
	float cosTheta = Frame::cosTheta(d);

	/* Check if the direction points behind the camera */
	if (cosTheta <= 0)
		return 0.0f;

	/* Compute the position on the plane at distance 1 */
	float invCosTheta = 1.0f / cosTheta;
	float2 p = make_float2(d.x * invCosTheta, d.y * invCosTheta);

		/* Check if the point lies inside the chosen crop rectangle */
	if (!m_imageRect.Contains(make_float3(p,0)))
		return 0.0f;
	return invCosTheta * invCosTheta * invCosTheta * m_normalization;
}

Spectrum e_PerspectiveCamera::sampleRay(Ray &ray, const float2 &pixelSample, const float2 &apertureSample) const
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

CUDA_FUNC_IN float PdfWtoA(const float pdfW, const float dist, const float cosThere)
{
    return pdfW * fabs(cosThere) / (dist * dist);
}

Spectrum e_PerspectiveCamera::sampleDirect(DirectSamplingRecord &dRec, const float2 &sample) const
{
	float3 refP = toWorldInverse * dRec.ref;

	/* Check if it is outside of the clip range */
	if (refP.z < m_fNearFarDepths.x || refP.z > m_fNearFarDepths.y) {
		dRec.pdf = 0.0f;
		return Spectrum(0.0f);
	}

	float3 screenSample = m_cameraToSample * refP;
	dRec.uv = make_float2(screenSample.x, screenSample.y);
	if (dRec.uv.x < 0 || dRec.uv.x  > 1 ||
		dRec.uv.y < 0 || dRec.uv.y > 1) {
		dRec.pdf = 0.0f;
		return Spectrum(0.0f);
	}

	dRec.uv.x *= m_resolution.x;
	dRec.uv.y *= m_resolution.y;

	float3 localD = refP;
	float dist	  = length(localD),
		  invDist = 1.0f / dist;
	localD *= invDist;

	dRec.p = toWorld.Translation();
	dRec.d = invDist * (dRec.p - dRec.ref);
	dRec.dist = dist;
	dRec.n = toWorld.Forward();
	dRec.pdf = 1;
	dRec.measure = EDiscrete;

	const float cosToCamera = dot(dRec.refN, dRec.d);
	const float cosAtCamera = Frame::cosTheta(-dRec.d);
	const float cameraPdfW = 1.f / (cosAtCamera * cosAtCamera * cosAtCamera) * m_normalization;
	const float cameraPdfA = PdfWtoA(cameraPdfW, dist, cosToCamera);
	//return cameraPdfA;
	return Spectrum(importance(localD)*invDist*invDist);
}

Spectrum e_PerspectiveCamera::sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const
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

bool e_PerspectiveCamera::getSamplePosition(const PositionSamplingRecord &pRec, const DirectionSamplingRecord &dRec, float2 &samplePosition) const
{
	float3 local = toWorldInverse.TransformNormal(dRec.d);

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

void e_ThinLensCamera::Update()
{
	e_SensorBase::Update();
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
	m_normalization = 1.0f / (m_imageRect.Size().x * m_imageRect.Size().y);
}

float e_ThinLensCamera::importance(const float3 &p, const float3 &d, float2* sample) const
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

	return m_normalization * invCosTheta * invCosTheta * invCosTheta;
}

Spectrum e_ThinLensCamera::sampleRay(Ray &ray, const float2 &pixelSample, const float2 &apertureSample) const
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

Spectrum e_ThinLensCamera::sampleDirect(DirectSamplingRecord &dRec, const float2 &sample) const
{
	float3 refP = toWorldInverse * dRec.ref;

	/* Check if it is outside of the clip range */
	if (refP.z < m_fNearFarDepths.x || refP.z > m_fNearFarDepths.y) {
		dRec.pdf = 0.0f;
		return Spectrum(0.0f);
	}

	/* Sample a position on the aperture (in local coordinates) */
	float2 tmp = Warp::squareToUniformDiskConcentric(sample) * m_apertureRadius;
	float3 apertureP = make_float3(tmp.x, tmp.y, 0);

	/* Compute the normalized direction vector from the
		aperture position to the reference point */
	float3 localD = (refP - apertureP);
	float dist = length(localD),
			invDist = 1.0f / dist;
	localD *= invDist;

	float value = importance(apertureP, localD, &dRec.uv);
	if (value == 0.0f) {
		dRec.pdf = 0.0f;
		return Spectrum(0.0f);
	}

	dRec.p = toWorld * apertureP;
	dRec.d = (dRec.p - dRec.ref) * invDist;
	dRec.dist = dist;
	dRec.n = toWorld.Forward();
	dRec.pdf = m_aperturePdf * dist*dist/(Frame::cosTheta(localD));
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

Spectrum e_ThinLensCamera::sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const
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

	float3 apertureP = toWorldInverse * pRec.p;

	/* Turn that into a normalized ray direction */
	float3 d = normalize(nearP - apertureP);
	dRec.d = toWorld.TransformNormal(d);
	dRec.measure = ESolidAngle;
	dRec.pdf = m_normalization / (d.z * d.z * d.z);

	return Spectrum(1.0f);
}

void e_OrthographicCamera::Update()
{
	e_SensorBase::Update();
	m_sampleToCamera = float4x4::Scale(make_float3(2, 2, 1)) * float4x4::Translate(make_float3(-1.0f, -1.0f, 0.0f)) * float4x4::Orthographic(screenScale.x, screenScale.y, m_fNearFarDepths.x, m_fNearFarDepths.y).Inverse();
	m_cameraToSample = m_sampleToCamera.Inverse();

	m_invSurfaceArea = 1.0f / (
		length(toWorld * m_sampleToCamera.Right()) *
		length(toWorld * m_sampleToCamera.Up()) );
	m_scale = length(toWorld.Forward());
}

Spectrum e_OrthographicCamera::sampleRay(Ray &ray, const float2 &pixelSample, const float2 &apertureSample) const
{
	float3 nearP = m_sampleToCamera * make_float3(
		pixelSample.x * m_invResolution.x,
		pixelSample.y * m_invResolution.y, 0.0f);

	ray = Ray(toWorld * make_float3(nearP.x, nearP.y, 0.0f), toWorld.Forward());

	return Spectrum(1.0f);
}

Spectrum e_OrthographicCamera::sampleDirect(DirectSamplingRecord &dRec, const float2 &) const
{
	dRec.n = toWorld.Forward();
	float scale = length(dRec.n);

	float3 localP = toWorldInverse * dRec.ref;
	localP.z *= scale;

	float3 sample = m_cameraToSample * localP;

	if (sample.x < 0 || sample.x > 1 || sample.y < 0 ||
		sample.y > 1 || sample.z < 0 || sample.z > 1) {
		dRec.pdf = 0.0f;
		return Spectrum(0.0f);
	}

	dRec.p = toWorld * make_float3(localP.x, localP.y, 0.0f);
	dRec.n /= scale;
	dRec.d = -dRec.n;
	dRec.dist = localP.z;
	dRec.uv = make_float2(sample.x * m_resolution.x,
						  sample.y * m_resolution.y);
	dRec.pdf = 1.0f;
	dRec.measure = EDiscrete;

	return Spectrum(m_invSurfaceArea);
}

Spectrum e_OrthographicCamera::samplePosition(PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const
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

bool e_OrthographicCamera::getSamplePosition(const PositionSamplingRecord &pRec, const DirectionSamplingRecord &dRec, float2 &samplePosition) const
{
	float3 localP = toWorldInverse * pRec.p;
	float3 sample = m_cameraToSample * localP;

	if (sample.x < 0 || sample.x > 1 || sample.y < 0 || sample.y > 1)
		return false;

	samplePosition = make_float2(sample.x * m_resolution.x,
		                    sample.y * m_resolution.y);
	return true;
}

void e_TelecentricCamera::Update()
{
	e_SensorBase::Update();
	m_sampleToCamera = float4x4::Scale(make_float3(2, 2, 1)) * float4x4::Translate(make_float3(-1.0f, -1.0f, 0.0f)) * float4x4::Orthographic(screenScale.x, screenScale.y, m_fNearFarDepths.x, m_fNearFarDepths.y).Inverse();
	m_cameraToSample = m_sampleToCamera.Inverse();

	m_normalization = 1.0f / (
		length(toWorld * m_sampleToCamera.Right()) *
		length(toWorld * m_sampleToCamera.Up()) );

	m_aperturePdf = 1.0f / (PI * m_apertureRadius * m_apertureRadius);
}

Spectrum e_TelecentricCamera::sampleRay(Ray &ray, const float2 &pixelSample, const float2 &apertureSample) const
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

Spectrum e_TelecentricCamera::sampleDirect(DirectSamplingRecord &dRec, const float2 &sample) const
{
	float f = m_focusDistance, apertureRadius = m_apertureRadius / screenScale.x;

	float3 localP = toWorldInverse * dRec.ref;

	float dist = localP.z;
	if (dist < m_fNearFarDepths.x || dist > m_fNearFarDepths.y) {
		dRec.pdf = 0.0f;
		return Spectrum(0.0f);
	}

	/* Circle of confusion */
	float radius = abs(localP.z - f) * apertureRadius/f;
	radius += apertureRadius;

	/* Sample the ray origin */
	float2 disk = Warp::squareToUniformDiskConcentric(sample);
	float3 diskP = make_float3(disk.x*radius+localP.x, disk.y*radius+localP.y, 0.0f);

	/* Compute the intersection with the focal plane */
	float3 localD = localP - diskP;
	float3 intersection = diskP + localD * (f/localD.z);

	/* Determine the associated sample coordinates */
	float3 uv = m_cameraToSample * intersection;
	if (uv.x < 0 || uv.x > 1 || uv.y < 0 || uv.y > 1) {
		dRec.pdf = 0.0f;
		return Spectrum(0.0f);
	}

	dRec.uv = make_float2(uv.x, uv.y);
	dRec.p = toWorld * diskP;
	dRec.n = toWorld.Forward();
	dRec.d = dRec.p - dRec.ref;
	dRec.dist = length(dRec.d);
	dRec.d /= dRec.dist;
	dRec.measure = ESolidAngle;

	dRec.pdf = dist*dist / (-dot(dRec.n, dRec.d)* PI * radius*radius);

	return Spectrum(m_normalization);
}

Spectrum e_TelecentricCamera::samplePosition(PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const
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

Spectrum e_TelecentricCamera::sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const
{
	float3 nearP = m_sampleToCamera * make_float3(sample.x, sample.y, 0.0f);

	/* Turn that into a normalized ray direction */
	float3 d = normalize(nearP);
	dRec.d = toWorld.TransformNormal(d);
	dRec.measure = ESolidAngle;
	dRec.pdf = m_normalization / (d.z * d.z * d.z);

	return Spectrum(1.0f);
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

void e_Sensor::SetToWorld(const float3& pos, const float3& _f)
{
	float3 f = normalize(_f);
	float3 r = normalize(cross(f, make_float3(0,1,0)));
	float3 u = normalize(cross(r, f));
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