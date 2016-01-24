#include <stdafx.h>
#include "Buffer.h"
#include "Light.h"
#include "Mesh.h"
#include "MIPMap.h"
#include <Math/Sampling.h>
#include "TriangleData.h"

namespace CudaTracerLib {

Spectrum PointLight::sampleRay(NormalizedT<Ray> &ray, const Vec2f &spatialSample, const Vec2f &directionalSample) const
{
	ray = NormalizedT<Ray>(lightPos, Warp::squareToUniformSphere(directionalSample));
	return m_intensity * (4 * PI);
}

Spectrum PointLight::sampleDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const
{
	dRec.p = lightPos;
	dRec.pdf = 1.0f;
	dRec.measure = EDiscrete;
	dRec.uv = Vec2f(0.5f);
	Vec3f dir = dRec.p - dRec.ref;
	dRec.dist = length(dir);
	float invDist = 1.0f / dRec.dist;
	dRec.d = NormalizedT<Vec3f>(dir * invDist);
	dRec.n = NormalizedT<Vec3f>(0.0f);
	dRec.pdf = 1;
	dRec.measure = EDiscrete;

	return m_intensity * (invDist * invDist);
}

Spectrum PointLight::samplePosition(PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const
{
	pRec.p = lightPos;
	pRec.n = NormalizedT<Vec3f>(0.0f);
	pRec.pdf = 1.0f;
	pRec.measure = EDiscrete;
	return m_intensity * (4 * PI);
}

Spectrum PointLight::sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const
{
	dRec.d = Warp::squareToUniformSphere(sample);
	dRec.pdf = INV_FOURPI;
	dRec.measure = ESolidAngle;
	return Spectrum(1.0f);
}

CUDA_FUNC_IN bool needsUVSample(const Texture& tex)
{
	return !tex.Is<ConstantTexture>();
}

Spectrum DiffuseLight::sampleRay(NormalizedT<Ray> &ray, const Vec2f &spatialSample, const Vec2f &directionalSample) const
{
	PositionSamplingRecord pRec;
	DifferentialGeometry dg;
	shapeSet.SamplePosition(pRec, spatialSample, &dg.uv[0]);
	dg.P = pRec.p;
	dg.bary = pRec.uv;
	auto local = m_bOrthogonal ? NormalizedT<Vec3f>(0.0f, 0.0f, 1.0f) : Warp::squareToCosineHemisphere(directionalSample);
	ray = NormalizedT<Ray>(pRec.p, Frame(pRec.n).toWorld(local));
	return m_rad_texture.Evaluate(dg) * PI * shapeSet.Area();
}

Spectrum DiffuseLight::eval(const Vec3f& p, const Frame& sys, const NormalizedT<Vec3f> &d) const
{
	if (dot(sys.n, d) <= 0 || (m_bOrthogonal && dot(d, sys.n) < 1 - DeltaEpsilon))
		return 0.0f;
	else
	{
		DifferentialGeometry dg;
		dg.P = p;
		dg.n = sys.n;
		dg.sys = sys;
		if (needsUVSample(m_rad_texture))
			if (!shapeSet.getPosition(p, &dg.bary, dg.uv))
				printf("Could not sample position in shape set!\n");
		return m_rad_texture.Evaluate(dg);
	}
}

Spectrum DiffuseLight::sampleDirect(DirectSamplingRecord &dRec, const Vec2f &_sample) const
{
	Vec2f sample = _sample, uv;
	float sc = 1;
	if (m_bOrthogonal)
	{
		sc = PI;
		//sample random triangle
		Vec3f p0, p1, p2;
		Vec2f uv0, uv1, uv2;
		shapeSet.sampleTriangle(p0, p1, p2, uv0, uv1, uv2, dRec.pdf, sample.x);
		const NormalizedT<Vec3f> n = normalize(cross(p1 - p0, p2 - p0));
		float lambda = dot(p0, n) - dot(dRec.ref, n);// := (p_0 * n - p * n) / (n * n)
		dRec.p = dRec.ref + lambda * n;
		bool inTriangle = MonteCarlo::Barycentric(dRec.p, p0, p1, p2, dRec.uv.x, dRec.uv.y);
		if (!inTriangle)
		{
			dRec.pdf = 0.0f;
			return Spectrum(0.0f);
		}
		dRec.n = n;
		dRec.pdf = 1.0f / float(shapeSet.numTriangles());
		dRec.measure = EArea;
		uv = dRec.uv.x * uv0 + dRec.uv.y * uv1 + (1 - dRec.uv.x - dRec.uv.y) * uv2;
	}
	else shapeSet.SamplePosition(dRec, sample, &uv);
	Vec3f dir = dRec.p - dRec.ref;
	float distSquared = dir.lenSqr();
	dRec.dist = math::sqrt(distSquared);
	dRec.d = NormalizedT<Vec3f>(dir / dRec.dist);
	float dp = absdot(dRec.d, dRec.n);
	if (!m_bOrthogonal)
	{
		dRec.pdf *= dp != 0 ? (distSquared / dp) : 0.0f;
		dRec.measure = ESolidAngle;
	}
	else dRec.measure = EDiscrete;

	if (dot(dRec.d, dRec.refN) >= 0 && dot(dRec.d, dRec.n) < 0 && dRec.pdf != 0)
	{
		DifferentialGeometry dg;
		dg.P = dRec.p;
		dg.uv[0] = uv;
		dg.bary = dRec.uv;
		return m_rad_texture.Evaluate(dg) / dRec.pdf * sc;
	}
	else
	{
		dRec.pdf = 0.0f;
		return Spectrum(0.0f);
	}
}

float DiffuseLight::pdfDirect(const DirectSamplingRecord &dRec) const
{
	if (dot(dRec.d, dRec.refN) >= 0 && dot(dRec.d, dRec.n) < 0)
	{
		if (m_bOrthogonal)
			return dRec.measure == EDiscrete ? shapeSet.PdfTriangle(dRec.p) : 0.0f;

		float pdfPos = shapeSet.PdfPosition();

		if (dRec.measure == ESolidAngle)
			return pdfPos * (dRec.dist * dRec.dist) / absdot(dRec.d, dRec.n);
		else if (dRec.measure == EArea)
			return pdfPos;
		else
			return 0.0f;
	}
	else {
		return 0.0f;
	}
}

Spectrum DiffuseLight::sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const
{
	NormalizedT<Vec3f> local = Warp::squareToCosineHemisphere(sample);
	dRec.d = m_bOrthogonal ? pRec.n : Frame(pRec.n).toWorld(local);
	dRec.pdf = m_bOrthogonal ? 1 : Warp::squareToCosineHemispherePdf(local);
	dRec.measure = ESolidAngle;
	return Spectrum(1.0f);
}

float DiffuseLight::pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
{
	float dp = dot(dRec.d, pRec.n);

	if (dRec.measure != ESolidAngle || dp < 0)
		dp = 0.0f;
	if (m_bOrthogonal && dp < 1 - DeltaEpsilon)
		return 0.0f;

	return m_bOrthogonal ? 1 : INV_PI * dp;
}

Spectrum DiffuseLight::evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
{
	float dp = dot(dRec.d, pRec.n);

	if (dRec.measure != ESolidAngle || dp < 0)
		dp = 0.0f;
	if (m_bOrthogonal && dp < 1 - DeltaEpsilon)
		return 0.0f;

	return m_bOrthogonal ? Spectrum(1.0f) : Spectrum(INV_PI * dp);
}

Spectrum DiffuseLight::evalPosition(const PositionSamplingRecord &pRec) const
{
	DifferentialGeometry dg;
	dg.P = pRec.p;
	if (needsUVSample(m_rad_texture))
		if (shapeSet.getPosition(dg.P, &dg.bary, &dg.uv[0]))
			printf("Could not sample position in shape set!\n");
	return m_rad_texture.Evaluate(dg) * PI;
}

Spectrum DiffuseLight::samplePosition(PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const
{
	DifferentialGeometry dg;
	shapeSet.SamplePosition(pRec, sample, &dg.uv[0]);
	dg.P = pRec.p;
	dg.bary = pRec.uv;
	return m_rad_texture.Evaluate(dg) * PI * shapeSet.Area();
}

void DistantLight::setEmit(const Spectrum& L)
{
	m_normalIrradiance = L;
	m_power = m_normalIrradiance / m_invSurfaceArea;
}

Spectrum DistantLight::sampleRay(NormalizedT<Ray> &ray, const Vec2f &spatialSample, const Vec2f &directionalSample) const
{
	Vec2f p = Warp::squareToUniformDiskConcentric(spatialSample);
	Vec3f perpOffset = ToWorld.toWorld(Vec3f(p.x, p.y, 0) * radius);
	auto d = ToWorld.toWorld(NormalizedT<Vec3f>(0.0f, 0.0f, 1.0f));
	ray = NormalizedT<Ray>(d * radius + perpOffset, d);
	return m_power;
}

Spectrum DistantLight::sampleDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const
{
	NormalizedT<Vec3f> d = ToWorld.toWorld(NormalizedT<Vec3f>(0.0f, 0.0f, 1.0f));
	Vec3f diskCenter = d * radius;

	float distance = dot(dRec.ref - diskCenter, d);
	if (distance < 0) {
		/* This can happen when doing bidirectional renderings
			involving environment maps and directional sources. Just
			return zero */
		return Spectrum(0.0f);
	}

	dRec.p = dRec.ref - distance * d;
	dRec.d = -d;
	dRec.n = d;
	dRec.dist = distance;

	dRec.pdf = 1.0f;
	dRec.measure = EDiscrete;
	return m_normalIrradiance;
}

Spectrum DistantLight::samplePosition(PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const
{
	Vec2f p = Warp::squareToUniformDiskConcentric(sample);

	Vec3f perpOffset = ToWorld.toWorld(Vec3f(p.x, p.y, 0) * radius);
	NormalizedT<Vec3f> d = ToWorld.toWorld(NormalizedT<Vec3f>(0.0f, 0.0f, 1.0f));

	pRec.p = d * radius + perpOffset;
	pRec.n = d;
	pRec.pdf = m_invSurfaceArea;
	pRec.measure = EArea;
	return m_power;
}

Spectrum DistantLight::sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const
{
	dRec.d = pRec.n;
	dRec.pdf = 1.0f;
	dRec.measure = EDiscrete;
	return Spectrum(1.0f);
}

SpotLight::SpotLight(Vec3f p, Vec3f t, Spectrum L, float width, float fall)
	: LightBase(true), m_intensity(L), m_cutoffAngle(math::Radians(width)), m_beamWidth(math::Radians(fall))
{
	m_cosBeamWidth = cosf(m_beamWidth);
	m_cosCutoffAngle = cosf(m_cutoffAngle);
	m_invTransitionWidth = 1.0f / (m_cutoffAngle - m_beamWidth);
	Position = p;
	Target = t;
	ToWorld = Frame((t - p).normalized());
}

Spectrum SpotLight::sampleRay(NormalizedT<Ray> &ray, const Vec2f &spatialSample, const Vec2f &directionalSample) const
{
	auto local = Warp::squareToUniformCone(m_cosCutoffAngle, directionalSample);
	ray = NormalizedT<Ray>(Position, ToWorld.toWorld(local).normalized());
	float dirPdf = Warp::squareToUniformConePdf(m_cosCutoffAngle);
	return m_intensity * falloffCurve(local.normalized()) / dirPdf;
}

Spectrum SpotLight::sampleDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const
{
	dRec.p = Position;
	dRec.pdf = 1.0f;
	dRec.measure = EDiscrete;
	dRec.uv = Vec2f(0.5f);
	Vec3f dir = dRec.p - dRec.ref;
	dRec.dist = length(dir);
	float invDist = 1.0f / dRec.dist;
	dRec.d = NormalizedT<Vec3f>(dir * invDist);
	dRec.n = NormalizedT<Vec3f>(0.0f);
	dRec.pdf = 1;
	dRec.measure = EDiscrete;

	return m_intensity * falloffCurve(ToWorld.toLocal(-dRec.d)) * (invDist * invDist);
}

Spectrum SpotLight::samplePosition(PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const
{
	pRec.p = Position;
	pRec.n = NormalizedT<Vec3f>(0.0f);
	pRec.pdf = 1.0f;
	pRec.measure = EDiscrete;
	return m_intensity * (4 * PI);
}

Spectrum SpotLight::sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const
{
	Vec3f d = Warp::squareToUniformCone(m_cosCutoffAngle, sample);
	dRec.d = ToWorld.toWorld(d).normalized();
	dRec.pdf = Warp::squareToUniformConePdf(m_cosCutoffAngle);
	dRec.measure = ESolidAngle;
	return evalDirection(dRec, pRec) / dRec.pdf;
}

Spectrum SpotLight::falloffCurve(const NormalizedT<Vec3f> &d) const
{
	const float cosTheta = Frame::cosTheta(d);

	if (cosTheta <= m_cosCutoffAngle)
		return Spectrum(0.0f);
	if (cosTheta >= m_cosBeamWidth)
		return 1.0f;
	return ((m_cutoffAngle - acosf(cosTheta)) * m_invTransitionWidth);
}

InfiniteLight::InfiniteLight(Stream<char>* a_Buffer, BufferReference<MIPMap, KernelMIPMap>& mip, const Spectrum& scale, const AABB* scenBox)
	: LightBase(false), radianceMap(mip->getKernelData()), m_scale(scale), m_pSceneBox(scenBox)
{
	m_size = Vec2f((float)radianceMap.m_uWidth, (float)radianceMap.m_uHeight);
	unsigned int nEntries = (unsigned int)(m_size.x + 1) * (unsigned int)m_size.y;
	StreamReference<char> m1 = a_Buffer->malloc_aligned<float>(nEntries * sizeof(float)), 
						  m2 = a_Buffer->malloc_aligned<float>((radianceMap.m_uHeight + 1) * sizeof(float)),
						  m3 = a_Buffer->malloc_aligned<float>(radianceMap.m_uHeight * sizeof(float));
	m_cdfCols = m1.AsVar<float>();
	m_cdfRows = m2.AsVar<float>();
	m_rowWeights = m3.AsVar<float>();
	unsigned int colPos = 0, rowPos = 0;
	float rowSum = 0.0f;
	m_cdfRows[rowPos++] = 0;
	for (int y = 0; y < m_size.y; ++y)
	{
		float colSum = 0;

		m_cdfCols[colPos++] = 0;
		for (int x = 0; x < m_size.x; ++x)
		{
			Spectrum value = radianceMap.Sample(0, x, y);

			colSum += value.getLuminance();
			m_cdfCols[colPos++] = (float)colSum;
		}

		float normalization = 1.0f / (float)colSum;
		for (int x = 1; x < m_size.x; ++x)
			m_cdfCols[colPos - x - 1] *= normalization;
		m_cdfCols[colPos - 1] = 1.0f;

		float weight = sinf((y + 0.5f) * PI / float(m_size.y));
		m_rowWeights[y] = weight;
		rowSum += colSum * weight;
		m_cdfRows[rowPos++] = (float)rowSum;
	}
	float normalization = 1.0f / (float)rowSum;
	for (int y = 1; y < m_size.y; ++y)
		m_cdfRows[rowPos - y - 1] *= normalization;
	m_cdfRows[rowPos - 1] = 1.0f;
	m_normalization = 1.0f / (rowSum * (2 * PI / m_size.x) * (PI / m_size.y));
	m_pixelSize = Vec2f(2 * PI / m_size.x, PI / m_size.y);
	m1.Invalidate(); m2.Invalidate(); m3.Invalidate();

	float lvl = 0.65f, qpdf;
	unsigned int INDEX = MonteCarlo::sampleReuse(m_cdfRows.operator->(), radianceMap.m_uHeight, lvl, qpdf);

	m_worldTransform = m_worldTransformInverse = float4x4::Identity();
}

Spectrum InfiniteLight::sampleRay(NormalizedT<Ray> &ray, const Vec2f &spatialSample, const Vec2f &directionalSample) const
{
	Vec3f d; Spectrum value; float pdf;
	internalSampleDirection(directionalSample, d, value, pdf);
	d = m_worldTransform.TransformDirection(-d);
	Vec2f offset = Warp::squareToUniformDiskConcentric(spatialSample);
	Vec3f perpOffset = Frame(d.normalized()).toWorld(Vec3f(offset.x, offset.y, 0.0f));
	ray = NormalizedT<Ray>(m_SceneCenter + (perpOffset - d) * m_SceneRadius, d.normalized());

	return value * PI * m_SceneRadius * m_SceneRadius / pdf;
}

Spectrum InfiniteLight::sampleDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const
{
	/* Sample a direction from the environment map */
	Spectrum value; Vec3f d; float pdf;
	internalSampleDirection(sample, d, value, pdf);
	d = m_worldTransform.TransformDirection(d);

	dRec.pdf = pdf;
	dRec.p = m_SceneCenter + d * m_SceneRadius;
	dRec.n = -normalize(d);
	dRec.dist = m_SceneRadius;
	dRec.d = d.normalized();
	dRec.measure = ESolidAngle;

	return value / pdf;
}

float InfiniteLight::pdfDirect(const DirectSamplingRecord &dRec) const
{
	float pdfSA = internalPdfDirection(m_worldTransformInverse.TransformDirection(dRec.d));

	if (dRec.measure == ESolidAngle)
		return pdfSA;
	else if (dRec.measure == EArea)
		return pdfSA * absdot(dRec.d, dRec.n) / (dRec.dist * dRec.dist);
	else
		return 0.0f;
}

Spectrum InfiniteLight::samplePosition(PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const
{
	NormalizedT<Vec3f> d = Warp::squareToUniformSphere(sample);

	pRec.p = m_SceneCenter + d * m_SceneRadius;
	pRec.n = -d;
	pRec.measure = EArea;
	pRec.pdf = m_invSurfaceArea;

	return Spectrum(m_power);
}

Spectrum InfiniteLight::sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const
{
	Spectrum value; Vec3f d; float pdf;
	internalSampleDirection(sample, d, value, pdf);

	dRec.measure = ESolidAngle;
	dRec.pdf = pdf;
	dRec.d = m_worldTransform.TransformDirection(-d).normalized();

	/* Be wary of roundoff errors */
	if (value.isZero() || pdf == 0)
		return Spectrum(0.0f);
	else
		return (value * m_normalization) / (pdf * m_scale);
}

Spectrum InfiniteLight::evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
{
	Vec3f v = m_worldTransformInverse.TransformDirection(-1.0f * dRec.d);

	/* Convert to latitude-longitude texture coordinates */
	Vec2f uv = Vec2f(
		atan2f(v.x, -v.z) * INV_TWOPI,
		math::safe_acos(v.y) * INV_PI
		);

	return radianceMap.Sample(uv, 0) * m_normalization;
}

void InfiniteLight::internalSampleDirection(Vec2f sample, Vec3f &d, Spectrum &value, float &pdf) const
{
	float qpdf;
	unsigned int	row = MonteCarlo::sampleReuse(m_cdfRows.operator->(), (unsigned int)m_size.y, sample.y, qpdf),
		col = MonteCarlo::sampleReuse(m_cdfCols.operator->() + row * (unsigned int)(m_size.x + 1), (unsigned int)m_size.x, sample.x, qpdf);

	/* Using the remaining bits of precision to shift the sample by an offset
		drawn from a tent function. This effectively creates a sampling strategy
		for a linearly interpolated environment map */

	Vec2f pos = Vec2f((float)col, (float)row) + Warp::squareToTent(sample);
	//Vec2f pos = sample * m_size;

	/* Bilinearly interpolate colors from the adjacent four neighbors */
	int xPos = math::clamp(math::Floor2Int(pos.x), 0, (int)(m_size.x - 1)), yPos = math::clamp(math::Floor2Int(pos.y), 0, (int)(m_size.y - 1));
	float dx1 = pos.x - xPos, dx2 = 1.0f - dx1,
		dy1 = pos.y - yPos, dy2 = 1.0f - dy1;

	Spectrum value1 = radianceMap.Sample(0, xPos, yPos) * dx2 * dy2
		+ radianceMap.Sample(0, xPos + 1, yPos) * dx1 * dy2;
	Spectrum value2 = radianceMap.Sample(0, xPos, yPos + 1) * dx2 * dy1
		+ radianceMap.Sample(0, xPos + 1, yPos + 1) * dx1 * dy1;

	/* Compute the final color and probability density of the sample */
	value = (value1 + value2) * m_scale;
	pdf = (value1.getLuminance() * m_rowWeights[(int)math::clamp((float)yPos, 0.0f, m_size.y - 1.0f)] +
		value2.getLuminance() * m_rowWeights[(int)math::clamp((float)(yPos + 1), 0.0f, m_size.y - 1.0f)]) * m_normalization;

	/* Turn into a proper direction on the sphere */
	float sinPhi, cosPhi, sinTheta, cosTheta;
	sincos(m_pixelSize.x * (pos.x + 0.5f), &sinPhi, &cosPhi);
	sincos(m_pixelSize.y * (pos.y + 0.5f), &sinTheta, &cosTheta);

	d = Vec3f(sinPhi*sinTheta, cosTheta, -cosPhi*sinTheta);
	pdf /= max(math::abs(sinTheta), EPSILON);
}

float InfiniteLight::internalPdfDirection(const Vec3f &d) const
{
	Vec2f uv = Vec2f(
		atan2f(d.x, -d.z) * INV_TWOPI,
		math::safe_acos(d.y) * INV_PI
		);
	float u = uv.x * m_size.x - 0.5f, v = uv.y * m_size.y - 0.5f;
	int xPos = math::Floor2Int(u), yPos = math::Floor2Int(v);
	float dx1 = u - xPos, dx2 = 1.0f - dx1,
		dy1 = v - yPos, dy2 = 1.0f - dy1;
	Spectrum value1 = radianceMap.Sample(0, xPos, yPos) * dx2 * dy2
		+ radianceMap.Sample(0, xPos + 1, yPos) * dx1 * dy2;
	Spectrum value2 = radianceMap.Sample(0, xPos, yPos + 1) * dx2 * dy1
		+ radianceMap.Sample(0, xPos + 1, yPos + 1) * dx1 * dy1;
	float sinTheta = math::safe_sqrt(1 - d.y*d.y);
	return (value1.getLuminance() * m_rowWeights[math::clamp(yPos, 0, (int)m_size.y - 1)] +
		value2.getLuminance() * m_rowWeights[math::clamp(yPos + 1, 0, (int)m_size.y - 1)])
		* m_normalization / max(math::abs(sinTheta), EPSILON);
}

Spectrum InfiniteLight::evalEnvironment(const Ray &ray) const
{
	Vec3f v = normalize(m_worldTransformInverse.TransformDirection(ray.dir()));

	/* Convert to latitude-longitude texture coordinates */
	Vec2f uv = Vec2f(
		atan2f(v.x, -v.z) * INV_TWOPI,
		math::safe_acos(v.y) * INV_PI
		);

	Spectrum value = radianceMap.Sample(uv, 0);

	return value * m_scale;
}

Spectrum InfiniteLight::evalEnvironment(const Ray &ray, const Ray& rX, const Ray& rY) const
{
	Vec3f v = normalize(m_worldTransformInverse.TransformDirection(ray.dir()));

	/* Convert to latitude-longitude texture coordinates */
	Vec2f uv = Vec2f(
		atan2f(v.x, -v.z) * INV_TWOPI,
		math::safe_acos(v.y) * INV_PI
		);

	Vec3f  dvdx = rX.dir() - v,
		dvdy = rY.dir() - v;

	float	t1 = INV_TWOPI / (v.x*v.x + v.z*v.z),
		t2 = -INV_PI / max(math::safe_sqrt(1.0f - v.y*v.y), 1e-4f);

	Vec2f	dudx = Vec2f(t1 * (dvdx.z*v.x - dvdx.x*v.z), t2 * dvdx.y),
		dudy = Vec2f(t1 * (dvdy.z*v.x - dvdy.x*v.z), t2 * dvdy.y);

	Spectrum value = radianceMap.eval(uv, dudx, dudy);

	return value * m_scale;
}

}
