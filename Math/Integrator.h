#pragma once

#include "MathFunc.h"
#include "Vector.h"

//Implementation copied from Mitsuba.

namespace CudaTracerLib {

#define m_alpha (math::sqrt(2.0f/3.0f))
#define m_beta ((1.0f/math::sqrt(5.0f)))
#define m_x1 (0.94288241569547971906f)
#define m_x2 (0.64185334234578130578f)
#define m_x3 (0.23638319966214988028f)

CUDA_DEVICE CUDA_HOST float legendreP(int l, float x);

CUDA_DEVICE CUDA_HOST float legendreP(int l, int m, float x);

CUDA_DEVICE CUDA_HOST Vec2f legendrePD(int l, float x);

CUDA_DEVICE CUDA_HOST Vec2f legendreQ(int l, float x);

CUDA_DEVICE CUDA_HOST void gaussLegendre(int n, float *nodes, float *weights);

CUDA_DEVICE CUDA_HOST void gaussLobatto(int n, float *nodes, float *weights);

class GaussLobattoIntegrator
{
	float m_absError, m_relError;
	size_t m_maxEvals;
	bool m_useConvergenceEstimate;
	bool m_warn;
public:
	CUDA_FUNC_IN GaussLobattoIntegrator(size_t maxEvals,
		float absError = 0,
		float relError = 0,
		bool useConvergenceEstimate = true,
		bool warn = true)
		: m_absError(absError),
		m_relError(relError),
		m_maxEvals(maxEvals),
		m_useConvergenceEstimate(useConvergenceEstimate),
		m_warn(warn)
	{

	}

	template<typename Integrand> CUDA_FUNC_IN float integrate(const Integrand &f, float a, float b, size_t *_evals = NULL) const
	{
		float factor = 1;
		size_t evals = 0;
		if (a == b) {
			return 0;
		}
		else if (b < a) {
			swapk(a, b);
			factor = -1;
		}
		const float absTolerance = calculateAbsTolerance(f, a, b, evals);
		evals += 2;
		float result = factor * adaptiveGaussLobattoStep(f, a, b, f(a), f(b), absTolerance, evals);
		if (_evals)
			*_evals = evals;
		return result;
	}
private:
	template<typename Integrand> CUDA_FUNC_IN float adaptiveGaussLobattoStep(const Integrand& f, float a, float b, float fa, float fb, float acc, size_t &evals) const
	{
		const float h = (b - a) / 2;
		const float m = (a + b) / 2;

		const float mll = m - m_alpha*h;
		const float ml = m - m_beta*h;
		const float mr = m + m_beta*h;
		const float mrr = m + m_alpha*h;

		const float fmll = f(mll);
		const float fml = f(ml);
		const float fm = f(m);
		const float fmr = f(mr);
		const float fmrr = f(mrr);

		const float integral2 = (h / 6)*(fa + fb + 5 * (fml + fmr));
		const float integral1 = (h / 1470)*(77 * (fa + fb)
			+ 432 * (fmll + fmrr) + 625 * (fml + fmr) + 672 * fm);

		evals += 5;

		if (evals >= m_maxEvals)
			return integral1;

		float dist = acc + (integral1 - integral2);
		if (dist == acc || mll <= a || b <= mrr) {
			return integral1;
		}
		else {
			return  adaptiveGaussLobattoStep(f, a, mll, fa, fmll, acc, evals)
				+ adaptiveGaussLobattoStep(f, mll, ml, fmll, fml, acc, evals)
				+ adaptiveGaussLobattoStep(f, ml, m, fml, fm, acc, evals)
				+ adaptiveGaussLobattoStep(f, m, mr, fm, fmr, acc, evals)
				+ adaptiveGaussLobattoStep(f, mr, mrr, fmr, fmrr, acc, evals)
				+ adaptiveGaussLobattoStep(f, mrr, b, fmrr, fb, acc, evals);
		}
	}

	template<typename Integrand> CUDA_FUNC_IN float calculateAbsTolerance(const Integrand& f, float a, float b, size_t &evals) const
	{
		const float m = (a + b) / 2;
		const float h = (b - a) / 2;
		const float y1 = f(a);
		const float y3 = f(m - m_alpha*h);
		const float y5 = f(m - m_beta*h);
		const float y7 = f(m);
		const float y9 = f(m + m_beta*h);
		const float y11 = f(m + m_alpha*h);
		const float y13 = f(b);

		float acc = h*(0.0158271919734801831f*(y1 + y13)
			+ 0.0942738402188500455f*(f(m - m_x1*h) + f(m + m_x1*h))
			+ 0.1550719873365853963f*(y3 + y11)
			+ 0.1888215739601824544f*(f(m - m_x2*h) + f(m + m_x2*h))
			+ 0.1997734052268585268f*(y5 + y9)
			+ 0.2249264653333395270f*(f(m - m_x3*h) + f(m + m_x3*h))
			+ 0.2426110719014077338f*y7);
		evals += 13;

		float r = 1.0;
		if (m_useConvergenceEstimate) {
			const float integral2 = (h / 6)*(y1 + y13 + 5 * (y5 + y9));
			const float integral1 = (h / 1470)*
				(77 * (y1 + y13) + 432 * (y3 + y11) + 625 * (y5 + y9) + 672 * y7);

			if (math::abs(integral2 - acc) != 0.0)
				r = math::abs(integral1 - acc) / math::abs(integral2 - acc);
			if (r == 0.0 || r > 1.0)
				r = 1.0;
		}

		float result = FLT_MAX;

		if (m_relError != 0 && acc != 0)
			result = acc * max(m_relError,
			FLT_EPSILON)
			/ (r*FLT_EPSILON);

		if (m_absError != 0)
			result = min(result, m_absError
			/ (r*FLT_EPSILON));

		return result;
	}
};

}