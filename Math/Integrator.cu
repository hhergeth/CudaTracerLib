#include "Integrator.h"
#include "float4x4.h"

float4x4 float4x4::inverse() const
{
	float4x4 Q = *this;
	float m00 = Q(0,0), m01 = Q(0,1), m02 = Q(0,2), m03 = Q(0,3);
	float m10 = Q(1,0), m11 = Q(1,1), m12 = Q(1,2), m13 = Q(1,3);
	float m20 = Q(2,0), m21 = Q(2,1), m22 = Q(2,2), m23 = Q(2,3);
	float m30 = Q(3,0), m31 = Q(3,1), m32 = Q(3,2), m33 = Q(3,3);

	float v0 = m20 * m31 - m21 * m30;
	float v1 = m20 * m32 - m22 * m30;
	float v2 = m20 * m33 - m23 * m30;
	float v3 = m21 * m32 - m22 * m31;
	float v4 = m21 * m33 - m23 * m31;
	float v5 = m22 * m33 - m23 * m32;

	float t00 = +(v5 * m11 - v4 * m12 + v3 * m13);
	float t10 = -(v5 * m10 - v2 * m12 + v1 * m13);
	float t20 = +(v4 * m10 - v2 * m11 + v0 * m13);
	float t30 = -(v3 * m10 - v1 * m11 + v0 * m12);

	float invDet = 1 / (t00 * m00 + t10 * m01 + t20 * m02 + t30 * m03);

	float d00 = t00 * invDet;
	float d10 = t10 * invDet;
	float d20 = t20 * invDet;
	float d30 = t30 * invDet;

	float d01 = -(v5 * m01 - v4 * m02 + v3 * m03) * invDet;
	float d11 = +(v5 * m00 - v2 * m02 + v1 * m03) * invDet;
	float d21 = -(v4 * m00 - v2 * m01 + v0 * m03) * invDet;
	float d31 = +(v3 * m00 - v1 * m01 + v0 * m02) * invDet;

	v0 = m10 * m31 - m11 * m30;
	v1 = m10 * m32 - m12 * m30;
	v2 = m10 * m33 - m13 * m30;
	v3 = m11 * m32 - m12 * m31;
	v4 = m11 * m33 - m13 * m31;
	v5 = m12 * m33 - m13 * m32;

	float d02 = +(v5 * m01 - v4 * m02 + v3 * m03) * invDet;
	float d12 = -(v5 * m00 - v2 * m02 + v1 * m03) * invDet;
	float d22 = +(v4 * m00 - v2 * m01 + v0 * m03) * invDet;
	float d32 = -(v3 * m00 - v1 * m01 + v0 * m02) * invDet;

	v0 = m21 * m10 - m20 * m11;
	v1 = m22 * m10 - m20 * m12;
	v2 = m23 * m10 - m20 * m13;
	v3 = m22 * m11 - m21 * m12;
	v4 = m23 * m11 - m21 * m13;
	v5 = m23 * m12 - m22 * m13;

	float d03 = -(v5 * m01 - v4 * m02 + v3 * m03) * invDet;
	float d13 = +(v5 * m00 - v2 * m02 + v1 * m03) * invDet;
	float d23 = -(v4 * m00 - v2 * m01 + v0 * m03) * invDet;
	float d33 = +(v3 * m00 - v1 * m01 + v0 * m02) * invDet;

	return float4x4::As(
		d00, d01, d02, d03,
		d10, d11, d12, d13,
		d20, d21, d22, d23,
		d30, d31, d32, d33);
	/*
	float4x4 r;
	float* inv = r.data;
	const float* m = data;
	inv[0] = m[5] * m[10] * m[15] -
		m[5] * m[11] * m[14] -
		m[9] * m[6] * m[15] +
		m[9] * m[7] * m[14] +
		m[13] * m[6] * m[11] -
		m[13] * m[7] * m[10];

	inv[4] = -m[4] * m[10] * m[15] +
		m[4] * m[11] * m[14] +
		m[8] * m[6] * m[15] -
		m[8] * m[7] * m[14] -
		m[12] * m[6] * m[11] +
		m[12] * m[7] * m[10];

	inv[8] = m[4] * m[9] * m[15] -
		m[4] * m[11] * m[13] -
		m[8] * m[5] * m[15] +
		m[8] * m[7] * m[13] +
		m[12] * m[5] * m[11] -
		m[12] * m[7] * m[9];

	inv[12] = -m[4] * m[9] * m[14] +
		m[4] * m[10] * m[13] +
		m[8] * m[5] * m[14] -
		m[8] * m[6] * m[13] -
		m[12] * m[5] * m[10] +
		m[12] * m[6] * m[9];

	inv[1] = -m[1] * m[10] * m[15] +
		m[1] * m[11] * m[14] +
		m[9] * m[2] * m[15] -
		m[9] * m[3] * m[14] -
		m[13] * m[2] * m[11] +
		m[13] * m[3] * m[10];

	inv[5] = m[0] * m[10] * m[15] -
		m[0] * m[11] * m[14] -
		m[8] * m[2] * m[15] +
		m[8] * m[3] * m[14] +
		m[12] * m[2] * m[11] -
		m[12] * m[3] * m[10];

	inv[9] = -m[0] * m[9] * m[15] +
		m[0] * m[11] * m[13] +
		m[8] * m[1] * m[15] -
		m[8] * m[3] * m[13] -
		m[12] * m[1] * m[11] +
		m[12] * m[3] * m[9];

	inv[13] = m[0] * m[9] * m[14] -
		m[0] * m[10] * m[13] -
		m[8] * m[1] * m[14] +
		m[8] * m[2] * m[13] +
		m[12] * m[1] * m[10] -
		m[12] * m[2] * m[9];

	inv[2] = m[1] * m[6] * m[15] -
		m[1] * m[7] * m[14] -
		m[5] * m[2] * m[15] +
		m[5] * m[3] * m[14] +
		m[13] * m[2] * m[7] -
		m[13] * m[3] * m[6];

	inv[6] = -m[0] * m[6] * m[15] +
		m[0] * m[7] * m[14] +
		m[4] * m[2] * m[15] -
		m[4] * m[3] * m[14] -
		m[12] * m[2] * m[7] +
		m[12] * m[3] * m[6];

	inv[10] = m[0] * m[5] * m[15] -
		m[0] * m[7] * m[13] -
		m[4] * m[1] * m[15] +
		m[4] * m[3] * m[13] +
		m[12] * m[1] * m[7] -
		m[12] * m[3] * m[5];

	inv[14] = -m[0] * m[5] * m[14] +
		m[0] * m[6] * m[13] +
		m[4] * m[1] * m[14] -
		m[4] * m[2] * m[13] -
		m[12] * m[1] * m[6] +
		m[12] * m[2] * m[5];

	inv[3] = -m[1] * m[6] * m[11] +
		m[1] * m[7] * m[10] +
		m[5] * m[2] * m[11] -
		m[5] * m[3] * m[10] -
		m[9] * m[2] * m[7] +
		m[9] * m[3] * m[6];

	inv[7] = m[0] * m[6] * m[11] -
		m[0] * m[7] * m[10] -
		m[4] * m[2] * m[11] +
		m[4] * m[3] * m[10] +
		m[8] * m[2] * m[7] -
		m[8] * m[3] * m[6];

	inv[11] = -m[0] * m[5] * m[11] +
		m[0] * m[7] * m[9] +
		m[4] * m[1] * m[11] -
		m[4] * m[3] * m[9] -
		m[8] * m[1] * m[7] +
		m[8] * m[3] * m[5];

	inv[15] = m[0] * m[5] * m[10] -
		m[0] * m[6] * m[9] -
		m[4] * m[1] * m[10] +
		m[4] * m[2] * m[9] +
		m[8] * m[1] * m[6] -
		m[8] * m[2] * m[5];

	float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

	if (det == 0)
		return float4x4::Identity();

	det = 1.0 / det;

	for (int i = 0; i < 16; i++)
		inv[i] = inv[i] * det;
	return r;*/
}

float legendreP(int l, float x) {

	if (l == 0) {
		return (float) 1.0f;
	} else if (l == 1) {
		return x;
	} else {
		float Lppred = 1.0, Lpred = x, Lcur = 0.0;

		for (int k = 2; k <= l; ++k) {
			Lcur = ((2*k-1) * x * Lpred - (k - 1) * Lppred) / k;
			Lppred = Lpred; Lpred = Lcur;
		}

		return (float) Lcur;
	}
}

float2 legendrePD(int l, float x_) {

	if (l == 0) {
		return make_float2((float) 1.0f, (float) 0.0f);
	} else if (l == 1) {
		return make_float2(x_, (float) 1.0f);
	} else {
		float x = (float) x_;
		float Lppred = 1.0, Lpred = x, Lcur = 0.0,
		       Dppred = 0.0, Dpred = 1.0, Dcur = 0.0;

		for (int k = 2; k <= l; ++k) {
			Lcur = ((2*k-1) * x * Lpred - (k - 1) * Lppred) / k;
			Dcur = Dppred + (2*k-1) * Lpred;
			Lppred = Lpred; Lpred = Lcur;
			Dppred = Dpred; Dpred = Dcur;
		}

		return make_float2((float) Lcur, (float) Dcur);
	}
}

float legendreP(int l, int m, float x) {
	float p_mm = 1;

	if (m > 0) {
		float somx2 = std::sqrt((1 - x) * (1 + x));
		float fact = 1;
		for (int i=1; i<=m; i++) {
			p_mm *= (-fact) * somx2;
			fact += 2;
		}
	}

	if (l == m)
		return (float) p_mm;

	float p_mmp1 = x * (2*m + 1) * p_mm;
	if (l == m+1)
		return (float) p_mmp1;

	float p_ll = 0;
	for (int ll=m+2; ll <= l; ++ll) {
		p_ll = ((2*ll-1)*x*p_mmp1 - (ll+m-1) * p_mm) / (ll-m);
		p_mm = p_mmp1;
		p_mmp1 = p_ll;
	}

	return (float) p_ll;
}

float2 legendreQ(int l, float x)
{
	if (l == 1) {
		return make_float2(0.5 * (3*x*x-1) - 1, 3*x);
	} else {
		/* Evaluate the recurrence in double precision */
		float Lppred = 1.0, Lpred = x, Lcur = 0.0,
		       Dppred = 0.0, Dpred = 1.0, Dcur = 0.0;

		for (int k = 2; k <= l; ++k) {
			Lcur = ((2*k-1) * x * Lpred - (k-1) * Lppred) / k;
			Dcur = Dppred + (2*k-1) * Lpred;
			Lppred = Lpred; Lpred = Lcur;
			Dppred = Dpred; Dpred = Dcur;
		}

		float Lnext = ((2*l+1) * x * Lpred - l * Lppred) / (l+1);
		float Dnext = Dppred + (2*l+1) * Lpred;

		return make_float2(Lnext - Lppred, Dnext - Dppred);
	}
}

void gaussLegendre(int n, float *nodes, float *weights) {
	if (n-- < 1)
		return;

	if (n == 0) {
		nodes[0] = 0;
		weights[0] = 2;
	} else if (n == 1) {
		nodes[0] = (float) -sqrtf(1.0/3.0);
		nodes[1] = -nodes[0];
		weights[0] = weights[1] = 1;
	}

	int m = (n+1)/2;
	for (int i=0; i<m; ++i) {
		/* Initial guess for this root using that of a Chebyshev polynomial */

		float x = -cosf((float) (2*i + 1) / (float) (2*n + 2) * PI);
		int it = 0;

		while (true) {
			if (++it > 20)
				return;

			/* Search for the interior roots of P_{n+1}(x) using Newton's method. */
			float2 L = legendrePD(n+1, x);
			float step = L.x / L.y;
			x -= step;

			if (abs(step) <= 4 * abs(x) * FLT_EPSILON)
				break;
		}

		float2 L = legendrePD(n+1, x);
		weights[i] = weights[n-i] = (float) (2.0f / ((1-x*x) * (L.y*L.y)));
		nodes[i] = (float) x; nodes[n-i] = (float) -x;
	}

	if ((n % 2) == 0) {
		float2 L = legendrePD(n+1, 0.0f);
		weights[n/2] = (float) (2.0f / (L.y*L.y));
		nodes[n/2] = 0;
	}
}

void gaussLobatto(int n, float *nodes, float *weights) {
	if (n-- < 2)
		return;

	nodes[0] = -1;
	nodes[n] =  1;
	weights[0] = weights[n] = (float) 2 / (float) (n * (n+1));

	int m = (n+1)/2;
	for (int i=1; i<m; ++i) {
		/* Initial guess for this root -- see "On the Legendre-Gauss-Lobatto Points
		   and Weights" by Seymor V. Parter, Journal of Sci. Comp., Vol. 14, 4, 1999 */

		float x = -cosf((i + 0.25f) * PI / n - 3/(8*n*PI * (i + 0.25f)));
		int it = 0;

		while (true) {
			if (++it > 20)
				return;

			/* Search for the interior roots of P_n'(x) using Newton's method. The same
			   roots are also shared by P_{n+1}-P_{n-1}, which is nicer to evaluate. */

			float2 Q = legendreQ(n, x);
			float step = Q.x / Q.y;
			x -= step;

			if (abs(step) <= 4 * abs(x) * FLT_EPSILON)
				break;
		}

		float Ln = legendreP(n, x);
		weights[i] = weights[n-i] = (float) (2.0f / ((n * (n+1)) * Ln * Ln));
		nodes[i] = (float) x; nodes[n-i] = (float) -x;
	}

	if ((n % 2) == 0) {
		float Ln = legendreP(n, 0.0);
		weights[n/2] = (float) (2.0f / ((n * (n+1)) * Ln * Ln));
		nodes[n/2] = 0.0;
	}
}