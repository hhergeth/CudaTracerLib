#pragma once

#include "..\MathTypes.h"
#include "e_Samples.h"

//this architecture and the implementations are completly copied from mitsuba!

struct e_SensorBase
{
public:
	float aspect;
	float2 m_resolution, m_invResolution;
	float4x4 toWorld;
public:
	e_SensorBase()
	{
		toWorld = float4x4::Identity();
	}
	CUDA_FUNC_IN void SetFilmData(int w, int h)
	{
		m_resolution = make_float2(w, h);
		m_invResolution = make_float2(1.0f) / m_resolution;
		aspect = m_resolution.x / m_resolution.y;
	}
	CUDA_FUNC_IN void SetToWorld(const float4x4& w)
	{
		toWorld = w;
	}
};

struct e_SphericalCamera : public e_SensorBase
{
	Spectrum sampleRay(Ray &ray, const float2 &pixelSample, const float2 &apertureSample)
	{
		float sinPhi, cosPhi, sinTheta, cosTheta;
		sincos(pixelSample.x * m_invResolution.x * 2 * PI, &sinPhi, &cosPhi);
		sincos(pixelSample.y * m_invResolution.y * PI, &sinTheta, &cosTheta);

		float3 d = make_float3(sinPhi*sinTheta, cosTheta, -cosPhi*sinTheta);
		ray = Ray(toWorld * make_float3(0), toWorld.TransformNormal(d));

		return Spectrum(1.0f);
	}

	Spectrum samplePosition(PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const
	{
		pRec.p = toWorld * make_float3(0);
		pRec.n = make_float3(0.0f);
		pRec.pdf = 1.0f;
		pRec.measure = EDiscrete;
		return Spectrum(1.0f);
	}

	Spectrum evalPosition(const PositionSamplingRecord &pRec) const
	{
		return Spectrum((pRec.measure == EDiscrete) ? 1.0f : 0.0f);
	}

	float pdfPosition(const PositionSamplingRecord &pRec) const
	{
		return (pRec.measure == EDiscrete) ? 1.0f : 0.0f;
	}

	Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const
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

	float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		if (dRec.measure != ESolidAngle)
			return 0.0f;

		float3 d = toWorld.Inverse().TransformNormal(dRec.d);
		float sinTheta = math::safe_sqrt(1-d.y*d.y);

		return 1 / (2 * PI * PI * MAX(sinTheta, EPSILON));
	}

	Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		if (dRec.measure != ESolidAngle)
			return Spectrum(0.0f);

		float3 d = toWorld.Inverse().TransformNormal(dRec.d);
		float sinTheta = math::safe_sqrt(1-d.y*d.y);

		return Spectrum(1 / (2 * PI * PI * MAX(sinTheta, EPSILON)));
	}

	bool getSamplePosition(const PositionSamplingRecord &pRec, const DirectionSamplingRecord &dRec, float2 &samplePosition) const
	{
		float3 d = normalize(toWorld.Inverse().TransformNormal(dRec.d));

		samplePosition = make_float2(
			math::modulo(atan2(d.x, -d.z) * INV_TWOPI, (float) 1) * m_resolution.x,
			math::safe_acos(d.y) * INV_PI * m_resolution.y
		);

		return true;
	}
};

struct a : public e_SensorBase
{

};

#define CAM_SIZE sizeof(e_SphericalCamera)

struct e_Sensor
{
private:
	CUDA_ALIGN(16) unsigned char Data[CAM_SIZE];
	unsigned int type;
public:
	enum ESensorFlags {
		EDeltaTime             = 0x010,
		ENeedsApertureSample   = 0x020,
		EProjectiveCamera      = 0x100,
		EPerspectiveCamera     = 0x200,
		EOrthographicCamera    = 0x400,
		EPositionSampleMapsToPixels  = 0x1000,
		EDirectionSampleMapsToPixels = 0x2000
	};

	CUDA_FUNC_IN void SetFilmData(int w, int h)
	{

	}

	CUDA_FUNC_IN void SetToWorld(const float4x4& w)
	{

	}

	//Spectrum sampleRay(Ray &ray, const float2 &samplePosition, const float2 &apertureSample);
	//Spectrum eval(const Intersection &its, const float3 &d, float2 &samplePos) const;
	//Spectrum samplePosition(PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const;
	//Spectrum evalPosition(const PositionSamplingRecord &pRec) const;
	//bool getSamplePosition(const PositionSamplingRecord &pRec, const DirectionSamplingRecord &dRec, float2 &position) const;

	STD_VIRTUAL_SET
};