#pragma once

#include "cutil_math.h"
#include "Frame.h"

CUDA_FUNC_IN float intervalToTent(float sample)
{
	float sign;

	if (sample < 0.5f) {
		sign = 1;
		sample *= 2;
	} else {
		sign = -1;
		sample = 2 * (sample - 0.5f);
	}

	return sign * (1 - sqrtf(sample));
}

class Warp
{
public:
	CUDA_FUNC_IN static float3 squareToUniformSphere(const float2 &sample)
	{
		float z = 1.0f - 2.0f * sample.y;
		float r = sqrtf(1.0f - z*z);
		float sinPhi, cosPhi;
		sincos(2.0f * PI * sample.x, &sinPhi, &cosPhi);
		return make_float3(r * cosPhi, r * sinPhi, z);
	}

	CUDA_FUNC_IN static float squareToUniformSpherePdf() { return INV_FOURPI; }

	CUDA_FUNC_IN static float3 squareToUniformHemisphere(const float2 &sample)
	{
		float z = sample.x;
		float tmp = sqrtf(1.0f - z*z);

		float sinPhi, cosPhi;
		sincos(2.0f * PI * sample.y, &sinPhi, &cosPhi);

		return make_float3(cosPhi * tmp, sinPhi * tmp, z);
	}

	CUDA_FUNC_IN static float squareToUniformHemispherePdf() { return INV_TWOPI; }

	CUDA_FUNC_IN static float3 squareToCosineHemisphere(const float2 &sample)
	{
		float2 p = Warp::squareToUniformDiskConcentric(sample);
		float z = sqrtf(1.0f - p.x*p.x - p.y*p.y);
		return make_float3(p.x, p.y, z);
	}

	CUDA_FUNC_IN static float squareToCosineHemispherePdf(const float3 &d)
		{ return INV_PI * Frame::cosTheta(d); }

	CUDA_FUNC_IN static float3 squareToUniformCone(float cosCutoff, const float2 &sample)
	{
		float cosTheta = (1-sample.x) + sample.x * cosCutoff;
		float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

		float sinPhi, cosPhi;
		sincos(2.0f * PI * sample.y, &sinPhi, &cosPhi);

		return make_float3(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
	}

	CUDA_FUNC_IN static float squareToUniformConePdf(float cosCutoff) {
		return INV_TWOPI / (1-cosCutoff);
	}

	CUDA_FUNC_IN static float2 squareToUniformDisk(const float2 &sample)
	{
		float r = sqrtf(sample.x);
		float sinPhi, cosPhi;
		sincos(2.0f * PI * sample.y, &sinPhi, &cosPhi);

		return make_float2(
			cosPhi * r,
			sinPhi * r
		);
	}

	CUDA_FUNC_IN static float squareToUniformDiskPdf() { return INV_PI; }

	CUDA_FUNC_IN static float2 squareToUniformDiskConcentric(const float2 &sample)
	{
		float r1 = 2.0f*sample.x - 1.0f;
		float r2 = 2.0f*sample.y - 1.0f;

		/* Modified concencric map code with less branching (by Dave Cline), see
		   http://psgraphics.blogspot.ch/2011/01/improved-code-for-concentric-map.html */
		float phi, r;
		if (r1 == 0 && r2 == 0) {
			r = phi = 0;
		} if (r1*r1 > r2*r2) {
			r = r1;
			phi = (PI/4.0f) * (r2/r1);
		} else {
			r = r2;
			phi = (PI/2.0f) - (r1/r2) * (PI/4.0f);
		}

		float cosPhi, sinPhi;
		sincos(phi, &sinPhi, &cosPhi);

		return make_float2(r * cosPhi, r * sinPhi);
	}

	CUDA_FUNC_IN static float2 uniformDiskToSquareConcentric(const float2 &p)
	{
		float r   = sqrtf(p.x * p.x + p.y * p.y),
		phi = atan2(p.y, p.x),
		a, b;

		if (phi < -PI/4) {
  			/* in range [-pi/4,7pi/4] */
			phi += 2*PI;
		}

		if (phi < PI/4) { /* region 1 */
			a = r;
			b = phi * a / (PI/4);
		} else if (phi < 3*PI/4) { /* region 2 */
			b = r;
			a = -(phi - PI/2) * b / (PI/4);
		} else if (phi < 5*PI/4) { /* region 3 */
			a = -r;
			b = (phi - PI) * a / (PI/4);
		} else { /* region 4 */
			b = -r;
			a = -(phi - 3*PI/2) * b / (PI/4);
		}

		return make_float2(0.5f * (a+1), 0.5f * (b+1));
	}

	CUDA_FUNC_IN static float squareToUniformDiskConcentricPdf() { return INV_PI; }

	CUDA_FUNC_IN static float2 squareToUniformTriangle(const float2 &sample)
	{
		float a = sqrtf(1.0f - sample.x);
		return make_float2(1 - a, a * sample.y);
	}

	CUDA_FUNC_IN static float squareToStdNormalPdf(const float2 &pos)
	{
		return INV_TWOPI * expf(-(pos.x*pos.x + pos.y*pos.y)/2.0f);
	}

	CUDA_FUNC_IN static float2 squareToTent(const float2 &sample)
	{
		return make_float2(
			intervalToTent(sample.x),
			intervalToTent(sample.y)
		);
	}

	CUDA_FUNC_IN static float intervalToNonuniformTent(float a, float b, float c, float sample)
	{
		float factor;

		if (sample * (c-a) < b-a) {
			factor = a-b;
			sample *= (a-c)/(a-b);
		} else {
			factor = c-b;
			sample = (a-c)/(b-c) * (sample - (a-b)/(a-c));
		}

		return b + factor * (1-sqrtf(sample));
	}
};