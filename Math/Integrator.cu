#include "Integrator.h"
#include "float4x4.h"

namespace CudaTracerLib {

float legendreP(int l, float x) {

	if (l == 0) {
		return (float) 1.0f;
	}
	else if (l == 1) {
		return x;
	}
	else {
		float Lppred = 1.0, Lpred = x, Lcur = 0.0;

		for (int k = 2; k <= l; ++k) {
			Lcur = ((2 * k - 1) * x * Lpred - (k - 1) * Lppred) / k;
			Lppred = Lpred; Lpred = Lcur;
		}

		return (float)Lcur;
	}
}

Vec2f legendrePD(int l, float x_) {

	if (l == 0) {
		return Vec2f((float) 1.0f, (float) 0.0f);
	}
	else if (l == 1) {
		return Vec2f(x_, (float) 1.0f);
	}
	else {
		float x = (float)x_;
		float Lppred = 1.0, Lpred = x, Lcur = 0.0,
			Dppred = 0.0, Dpred = 1.0, Dcur = 0.0;

		for (int k = 2; k <= l; ++k) {
			Lcur = ((2 * k - 1) * x * Lpred - (k - 1) * Lppred) / k;
			Dcur = Dppred + (2 * k - 1) * Lpred;
			Lppred = Lpred; Lpred = Lcur;
			Dppred = Dpred; Dpred = Dcur;
		}

		return Vec2f((float)Lcur, (float)Dcur);
	}
}

float legendreP(int l, int m, float x) {
	float p_mm = 1;

	if (m > 0) {
		float somx2 = sqrt((1 - x) * (1 + x));
		float fact = 1;
		for (int i = 1; i <= m; i++) {
			p_mm *= (-fact) * somx2;
			fact += 2;
		}
	}

	if (l == m)
		return (float)p_mm;

	float p_mmp1 = x * (2 * m + 1) * p_mm;
	if (l == m + 1)
		return (float)p_mmp1;

	float p_ll = 0;
	for (int ll = m + 2; ll <= l; ++ll) {
		p_ll = ((2 * ll - 1)*x*p_mmp1 - (ll + m - 1) * p_mm) / (ll - m);
		p_mm = p_mmp1;
		p_mmp1 = p_ll;
	}

	return (float)p_ll;
}

Vec2f legendreQ(int l, float x)
{
	if (l == 1) {
		return Vec2f(0.5 * (3 * x*x - 1) - 1, 3 * x);
	}
	else {
		/* Evaluate the recurrence in double precision */
		float Lppred = 1.0, Lpred = x, Lcur = 0.0,
			Dppred = 0.0, Dpred = 1.0, Dcur = 0.0;

		for (int k = 2; k <= l; ++k) {
			Lcur = ((2 * k - 1) * x * Lpred - (k - 1) * Lppred) / k;
			Dcur = Dppred + (2 * k - 1) * Lpred;
			Lppred = Lpred; Lpred = Lcur;
			Dppred = Dpred; Dpred = Dcur;
		}

		float Lnext = ((2 * l + 1) * x * Lpred - l * Lppred) / (l + 1);
		float Dnext = Dppred + (2 * l + 1) * Lpred;

		return Vec2f(Lnext - Lppred, Dnext - Dppred);
	}
}

void gaussLegendre(int n, float *nodes, float *weights) {
	if (n-- < 1)
		return;

	if (n == 0) {
		nodes[0] = 0;
		weights[0] = 2;
	}
	else if (n == 1) {
		nodes[0] = (float)-math::sqrt(1.0f / 3.0f);
		nodes[1] = -nodes[0];
		weights[0] = weights[1] = 1;
	}

	int m = (n + 1) / 2;
	for (int i = 0; i < m; ++i) {
		/* Initial guess for this root using that of a Chebyshev polynomial */

		float x = -cosf((float)(2 * i + 1) / (float)(2 * n + 2) * PI);
		int it = 0;

		while (true) {
			if (++it > 20)
				return;

			/* Search for the interior roots of P_{n+1}(x) using Newton's method. */
			Vec2f L = legendrePD(n + 1, x);
			float step = L.x / L.y;
			x -= step;

			if (math::abs(step) <= 4 * math::abs(x) * FLT_EPSILON)
				break;
		}

		Vec2f L = legendrePD(n + 1, x);
		weights[i] = weights[n - i] = (float)(2.0f / ((1 - x*x) * (L.y*L.y)));
		nodes[i] = (float)x; nodes[n - i] = (float)-x;
	}

	if ((n % 2) == 0) {
		Vec2f L = legendrePD(n + 1, 0.0f);
		weights[n / 2] = (float)(2.0f / (L.y*L.y));
		nodes[n / 2] = 0;
	}
}

void gaussLobatto(int n, float *nodes, float *weights) {
	if (n-- < 2)
		return;

	nodes[0] = -1;
	nodes[n] = 1;
	weights[0] = weights[n] = (float)2 / (float)(n * (n + 1));

	int m = (n + 1) / 2;
	for (int i = 1; i < m; ++i) {
		/* Initial guess for this root -- see "On the Legendre-Gauss-Lobatto Points
		   and Weights" by Seymor V. Parter, Journal of Sci. Comp., Vol. 14, 4, 1999 */

		float x = -cosf((i + 0.25f) * PI / n - 3 / (8 * n*PI * (i + 0.25f)));
		int it = 0;

		while (true) {
			if (++it > 20)
				return;

			/* Search for the interior roots of P_n'(x) using Newton's method. The same
			   roots are also shared by P_{n+1}-P_{n-1}, which is nicer to evaluate. */

			Vec2f Q = legendreQ(n, x);
			float step = Q.x / Q.y;
			x -= step;

			if (math::abs(step) <= 4 * math::abs(x) * FLT_EPSILON)
				break;
		}

		float Ln = legendreP(n, x);
		weights[i] = weights[n - i] = (float)(2.0f / ((n * (n + 1)) * Ln * Ln));
		nodes[i] = (float)x; nodes[n - i] = (float)-x;
	}

	if ((n % 2) == 0) {
		float Ln = legendreP(n, 0.0);
		weights[n / 2] = (float)(2.0f / ((n * (n + 1)) * Ln * Ln));
		nodes[n / 2] = 0.0;
	}
}

}