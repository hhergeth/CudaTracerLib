#include "Spline.h"
#include "..\Base\STL.h"

float Spline::evalCubicInterp1D(float x, const float *values, size_t size, float min, float max, bool extrapolate)
{
	/* Give up when given an out-of-range or NaN argument */
	if (!(x >= min && x <= max) && !extrapolate)
		return 0.0f;

	/* Transform 'x' so that knots lie at integer positions */
	float t = ((x - min) * (size - 1)) / (max - min);

	/* Find the index of the left knot in the queried subinterval, be
	   robust to cases where 't' lies exactly on the right endpoint */
	size_t k = MAX((size_t) 0, MIN((size_t) t, size - 2));

	float f0  = values[k],
	      f1  = values[k+1],
	      d0, d1;

	/* Approximate the derivatives */
	if (k > 0)
		d0 = 0.5f * (values[k+1] - values[k-1]);
	else
		d0 = values[k+1] - values[k];

	if (k + 2 < size)
		d1 = 0.5f * (values[k+2] - values[k]);
	else
		d1 = values[k+1] - values[k];

	/* Compute the relative position within the interval */
	t = t - (float) k;

	float t2 = t*t, t3 = t2*t;

	return
		( 2*t3 - 3*t2 + 1) * f0 +
		(-2*t3 + 3*t2)     * f1 +
		(   t3 - 2*t2 + t) * d0 +
		(   t3 - t2)       * d1;
}

float Spline::evalCubicInterp1DN(float x, const float *nodes, const float *values, size_t size, bool extrapolate)
{
	/* Give up when given an out-of-range or NaN argument */
	if (!(x >= nodes[0] && x <= nodes[size-1]) && !extrapolate)
		return 0.0f;

	size_t k = (size_t) MAX((ptrdiff_t) 0, MIN((ptrdiff_t) size - 2,
		STL_lower_bound(nodes, nodes + size, x) - nodes - 1));

	float f0       = values[k],
	      f1       = values[k+1],
	      width    = nodes[k+1] - nodes[k],
	      d0, d1;

	/* Approximate the derivatives */
	if (k > 0)
		d0 = width * (f1 - values[k-1]) / (nodes[k+1] - nodes[k-1]);
	else
		d0 = f1 - f0;

	if (k + 2 < size)
		d1 = width * (values[k+2] - f0) / (nodes[k+2] - nodes[k]);
	else
		d1 = f1 - f0;

	float t = (x - nodes[k]) / width;
	float t2 = t*t, t3 = t2*t;

	return
	    ( 2*t3 - 3*t2 + 1) * f0 +
	    (-2*t3 + 3*t2)     * f1 +
	    (   t3 - 2*t2 + t) * d0 +
	    (   t3 - t2)       * d1;
}

float Spline::integrateCubicInterp1D(size_t idx, const float *values, size_t size, float min, float max)
{
	float f0 = values[idx], f1 = values[idx+1], d0, d1;

	/* Approximate the derivatives */
	if (idx > 0)
		d0 = 0.5f * (values[idx+1] - values[idx-1]);
	else
		d0 = values[idx+1] - values[idx];

	if (idx + 2 < size)
		d1 = 0.5f * (values[idx+2] - values[idx]);
	else
		d1 = values[idx+1] - values[idx];

	return ((d0-d1) * (float) (1.0 / 12.0) + (f0+f1) * 0.5f) * (max-min) / (size - 1);
}

float Spline::integrateCubicInterp1DN(size_t idx, const float *nodes, const float *values, size_t size)
{
	float f0       = values[idx],
	      f1       = values[idx+1],
	      width    = nodes[idx+1] - nodes[idx],
	      d0, d1;

	/* Approximate the derivatives */
	if (idx > 0)
		d0 = width * (f1 - values[idx-1]) / (nodes[idx+1] - nodes[idx-1]);
	else
		d0 = f1 - f0;

	if (idx + 2 < size)
		d1 = width * (values[idx+2] - f0) / (nodes[idx+2] - nodes[idx]);
	else
		d1 = f1 - f0;

	return ((d0-d1) * (float) (1.0 / 12.0) + (f0+f1) * 0.5f) * width;
}

float Spline::sampleCubicInterp1D(size_t idx, float *values, size_t size, float min, float max, float sample, float *fval)
{
	float f0 = values[idx], f1 = values[idx+1], d0, d1;

	/* Approximate the derivatives */
	if (idx > 0)
		d0 = 0.5f * (values[idx+1] - values[idx-1]);
	else
		d0 = values[idx+1] - values[idx];

	if (idx + 2 < size)
		d1 = 0.5f * (values[idx+2] - values[idx]);
	else
		d1 = values[idx+1] - values[idx];

	/* Bracketing interval and starting guess */
	float a = 0, c = 1, b;

	if (f0 != f1) /* Importance sample linear interpolant */
		b = (f0-math::safe_sqrt(f0*f0 + sample * (f1*f1-f0*f0))) / (f0-f1);
	else
		b = sample;

	sample *= ((d0-d1) * (float) (1.0 / 12.0) + (f0+f1) * 0.5f);

	/* Invert CDF using Newton-Bisection */
	while (true) {
		if (!(b >= a && b <= c))
			b = 0.5f * (a + c);

		/* CDF and PDF in Horner form */
		float value = b*(f0 + b*(.5f*d0 + b*((float) (1.0f/3.0f) * (-2*d0-d1)
			+ f1 - f0 + b*(0.25f*(d0 + d1) + 0.5f * (f0 - f1))))) - sample;
		float deriv = f0 + b*(d0 + b*(-2*d0 - d1 + 3*(f1-f0) + b*(d0 + d1 + 2*(f0 - f1))));

		if (abs(value) < 1e-6f) {
			if (fval)
				*fval = deriv;
			return min + (idx+b) * (max-min) / (size-1);
		}

		if (value > 0)
			c = b;
		else
			a = b;

		b -= value / deriv;
	}
}

float Spline::sampleCubicInterp1DN(size_t idx, float *nodes, float *values, size_t size, float sample, float *fval)
{
	float f0       = values[idx],
	      f1       = values[idx+1],
	      width    = nodes[idx+1] - nodes[idx],
	      d0, d1;

	/* Approximate the derivatives */
	if (idx > 0)
		d0 = width * (f1 - values[idx-1]) / (nodes[idx+1] - nodes[idx-1]);
	else
		d0 = f1 - f0;

	if (idx + 2 < size)
		d1 = width * (values[idx+2] - f0) / (nodes[idx+2] - nodes[idx]);
	else
		d1 = f1 - f0;

	/* Bracketing interval and starting guess */
	float a = 0, c = 1, b;

	if (f0 != f1) /* Importance sample linear interpolant */
		b = (f0-math::safe_sqrt(f0*f0 + sample * (f1*f1-f0*f0))) / (f0-f1);
	else
		b = sample;

	sample *= ((d0-d1) * (float) (1.0 / 12.0) + (f0+f1) * 0.5f);

	/* Invert CDF using Newton-Bisection */
	while (true) {
		if (!(b >= a && b <= c))
			b = 0.5f * (a + c);

		/* CDF and PDF in Horner form */
		float value = b*(f0 + b*(.5f*d0 + b*((float) (1.0f/3.0f) * (-2*d0-d1)
			+ f1 - f0 + b*(0.25f*(d0 + d1) + 0.5f * (f0 - f1))))) - sample;
		float deriv = f0 + b*(d0 + b*(-2*d0 - d1 + 3*(f1-f0) + b*(d0 + d1 + 2*(f0 - f1))));

		if (abs(value) < 1e-6f) {
			if (fval)
				*fval = deriv;
			return nodes[idx] + width*b;
		}

		if (value > 0)
			c = b;
		else
			a = b;

		b -= value / deriv;
	}
}

float Spline::evalCubicInterp2D(const float2 &_p, const float *values, const uint2 &_size, const float2 &_min, const float2 &_max, bool extrapolate)
{
	const float* p = (float*)&_p, *min = (float*)&_min, *max = (float*)&_max;
	const unsigned int* size = (unsigned int*)&_size;

	float knotWeights[2][4];
	unsigned int knot[2];

	/* Compute interpolation weights separately for each dimension */
	for (int dim=0; dim<2; ++dim) {
		float *weights = knotWeights[dim];
		/* Give up when given an out-of-range or NaN argument */
		if (!(p[dim] >= min[dim] && p[dim] <= max[dim]) && !extrapolate)
			return 0.0f;

		/* Transform 'p' so that knots lie at integer positions */
		float t = ((p[dim] - min[dim]) * (size[dim] - 1))
			/ (max[dim]-min[dim]);

		/* Find the index of the left knot in the queried subinterval, be
		   robust to cases where 't' lies exactly on the right endpoint */
		knot[dim] = MIN((unsigned int) t, size[dim] - 2);

		/* Compute the relative position within the interval */
		t = t - (float) knot[dim];

		/* Compute node weights */
		float t2 = t*t, t3 = t2*t;
		weights[0] = 0.0f;
		weights[1] = 2*t3 - 3*t2 + 1;
		weights[2] = -2*t3 + 3*t2;
		weights[3] = 0.0f;

		/* Derivative weights */
		float d0 = t3 - 2*t2 + t,
			  d1 = t3 - t2;

		/* Turn derivative weights into node weights using
		   an appropriate chosen finite differences stencil */
		if (knot[dim] > 0) {
			weights[2] +=  0.5f * d0;
			weights[0] -=  0.5f * d0;
		} else {
			weights[2] += d0;
			weights[1] -= d0;
		}

		if (knot[dim] + 2 < size[dim]) {
			weights[3] += 0.5f * d1;
			weights[1] -= 0.5f * d1;
		} else {
			weights[2] += d1;
			weights[1] -= d1;
		}
	}

	float result = 0.0f;
	for (int y=-1; y<=2; ++y) {
		float wy = knotWeights[1][y+1];
		for (int x=-1; x<=2; ++x) {
			float wxy = knotWeights[0][x+1] * wy;

			if (wxy == 0)
				continue;

			size_t pos = (knot[1] + y) * size[0] + knot[0] + x;

			result += values[pos] * wxy;
		}
	}
	return result;
}

float Spline::evalCubicInterp2DN(const float2 &_p, const float **nodes_, const float *values, const uint2 &_size, bool extrapolate)
{
	const float* p = (float*)&_p;
	const unsigned int* size = (unsigned int*)&_size;

	float knotWeights[2][4];
	unsigned int knot[2];

	/* Compute interpolation weights separately for each dimension */
	for (int dim=0; dim<2; ++dim) {
		const float *nodes = nodes_[dim];
		float *weights = knotWeights[dim];

		/* Give up when given an out-of-range or NaN argument */
		if (!(p[dim] >= nodes[0] && p[dim] <= nodes[size[dim]-1]) && !extrapolate)
			return 0.0f;

		/* Find the index of the left knot in the queried subinterval, be
		   robust to cases where 't' lies exactly on the right endpoint */
		int k = MAX((ptrdiff_t) 0, MIN((ptrdiff_t) size[dim] - 2,
			STL_lower_bound(nodes, nodes + size[dim], p[dim]) - nodes - 1));
		knot[dim] = k;

		float width = nodes[k+1] - nodes[k];

		/* Compute the relative position within the interval */
		float t = (p[dim] - nodes[k]) / width,
			  t2 = t*t, t3 = t2*t;

		/* Compute node weights */
		weights[0] = 0.0f;
		weights[1] = 2*t3 - 3*t2 + 1;
		weights[2] = -2*t3 + 3*t2;
		weights[3] = 0.0f;

		/* Derivative weights */
		float d0 = t3 - 2*t2 + t, d1 = t3 - t2;

		/* Turn derivative weights into node weights using
		   an appropriate chosen finite differences stencil */
		if (k > 0) {
			float factor = width / (nodes[k+1]-nodes[k-1]);
			weights[2] += d0 * factor;
			weights[0] -= d0 * factor;
		} else {
			weights[2] += d0;
			weights[1] -= d0;
		}

		if (k + 2 < (int)size[dim]) {
			float factor = width / (nodes[k+2]-nodes[k]);
			weights[3] += d1 * factor;
			weights[1] -= d1 * factor;
		} else {
			weights[2] += d1;
			weights[1] -= d1;
		}
	}

	float result = 0.0f;
	for (int y=-1; y<=2; ++y) {
		float wy = knotWeights[1][y+1];
		for (int x=-1; x<=2; ++x) {
			float wxy = knotWeights[0][x+1] * wy;

			if (wxy == 0)
				continue;

			size_t pos = (knot[1] + y) * size[0] + knot[0] + x;

			result += values[pos] * wxy;
		}
	}
	return result;
}

float Spline::evalCubicInterp3D(const float3 &_p, const float *values, const uint3 &_size, const float3 &_min, const float3 &_max, bool extrapolate)
{
	const float* p = (float*)&_p, *min = (float*)&_min, *max = (float*)&_max;
	const unsigned int* size = (unsigned int*)&_size;

	float knotWeights[3][4];
	unsigned int knot[3];

	/* Compute interpolation weights separately for each dimension */
	for (int dim=0; dim<3; ++dim) {
		float *weights = knotWeights[dim];
		/* Give up when given an out-of-range or NaN argument */
		if (!(p[dim] >= min[dim] && p[dim] <= max[dim]) && !extrapolate)
			return 0.0f;

		/* Transform 'p' so that knots lie at integer positions */
		float t = ((p[dim] - min[dim]) * (size[dim] - 1))
			/ (max[dim]-min[dim]);

		/* Find the index of the left knot in the queried subinterval, be
		   robust to cases where 't' lies exactly on the right endpoint */
		knot[dim] = MIN((unsigned int) t, size[dim] - 2);

		/* Compute the relative position within the interval */
		t = t - (float) knot[dim];

		/* Compute node weights */
		float t2 = t*t, t3 = t2*t;
		weights[0] = 0.0f;
		weights[1] = 2*t3 - 3*t2 + 1;
		weights[2] = -2*t3 + 3*t2;
		weights[3] = 0.0f;

		/* Derivative weights */
		float d0 = t3 - 2*t2 + t,
			  d1 = t3 - t2;

		/* Turn derivative weights into node weights using
		   an appropriate chosen finite differences stencil */
		if (knot[dim] > 0) {
			weights[2] +=  0.5f * d0;
			weights[0] -=  0.5f * d0;
		} else {
			weights[2] += d0;
			weights[1] -= d0;
		}

		if (knot[dim] + 2 < size[dim]) {
			weights[3] += 0.5f * d1;
			weights[1] -= 0.5f * d1;
		} else {
			weights[2] += d1;
			weights[1] -= d1;
		}
	}

	float result = 0.0f;
	for (int z=-1; z<=2; ++z) {
		float wz = knotWeights[2][z+1];
		for (int y=-1; y<=2; ++y) {
			float wyz = knotWeights[1][y+1] * wz;
			for (int x=-1; x<=2; ++x) {
				float wxyz = knotWeights[0][x+1] * wyz;

				if (wxyz == 0)
					continue;

				size_t pos = ((knot[2] + z) * size[1] + (knot[1] + y))
					* size[0] + knot[0] + x;

				result += values[pos] * wxyz;
			}
		}
	}
	return result;
}

float Spline::evalCubicInterp3DN(const float3 &_p, const float **nodes_, const float *values, const uint3 &_size, bool extrapolate)
{
	const float* p = (float*)&_p;
	const unsigned int* size = (unsigned int*)&_size;

	float knotWeights[3][4];
	unsigned int knot[3];

	/* Compute interpolation weights separately for each dimension */
	for (int dim=0; dim<3; ++dim) {
		const float *nodes = nodes_[dim];
		float *weights = knotWeights[dim];

		/* Give up when given an out-of-range or NaN argument */
		if (!(p[dim] >= nodes[0] && p[dim] <= nodes[size[dim]-1]) && !extrapolate)
			return 0.0f;

		/* Find the index of the left knot in the queried subinterval, be
		   robust to cases where 't' lies exactly on the right endpoint */
		int k = MAX((ptrdiff_t) 0, MIN((ptrdiff_t) size[dim] - 2,
			STL_lower_bound(nodes, nodes + size[dim], p[dim]) - nodes - 1));
		knot[dim] = k;

		float width = nodes[k+1] - nodes[k];

		/* Compute the relative position within the interval */
		float t = (p[dim] - nodes[k]) / width,
			  t2 = t*t, t3 = t2*t;

		/* Compute node weights */
		weights[0] = 0.0f;
		weights[1] = 2*t3 - 3*t2 + 1;
		weights[2] = -2*t3 + 3*t2;
		weights[3] = 0.0f;

		/* Derivative weights */
		float d0 = t3 - 2*t2 + t, d1 = t3 - t2;

		/* Turn derivative weights into node weights using
		   an appropriate chosen finite differences stencil */
		if (k > 0) {
			float factor = width / (nodes[k+1]-nodes[k-1]);
			weights[2] += d0 * factor;
			weights[0] -= d0 * factor;
		} else {
			weights[2] += d0;
			weights[1] -= d0;
		}

		if (k + 2 < (int)size[dim]) {
			float factor = width / (nodes[k+2]-nodes[k]);
			weights[3] += d1 * factor;
			weights[1] -= d1 * factor;
		} else {
			weights[2] += d1;
			weights[1] -= d1;
		}
	}

	float result = 0.0f;
	for (int z=-1; z<=2; ++z) {
		float wz = knotWeights[2][z+1];
		for (int y=-1; y<=2; ++y) {
			float wyz = knotWeights[1][y+1] * wz;
			for (int x=-1; x<=2; ++x) {
				float wxyz = knotWeights[0][x+1] * wyz;

				if (wxyz == 0)
					continue;

				size_t pos = ((knot[2] + z) * size[1] + (knot[1] + y))
					* size[0] + knot[0] + x;

				result += values[pos] * wxyz;
			}
		}
	}
	return result;
}