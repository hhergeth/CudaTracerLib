#pragma once
#include "qMatrixWrapper.h"
#include "Frame.h"

namespace CudaTracerLib {

template<typename T, int M, int N> qMatrix<T, 1, N> ddot_dx(const qMatrix<T, M, 1>& a, const qMatrix<T, M, N>& da_dx, const qMatrix<T, M, 1>& b, const qMatrix<T, M, N>& db_dx)
{
	return b.transpose() * da_dx + a.transpose() * db_dx;
}

template<typename T, int M, int N> qMatrix<T, 1, N> ddot_dx(const qMatrix<T, M, 1>& a, const qMatrix<T, M, N>& da_dx)
{
	return ddot_dx(a, da_dx, a, da_dx);
}

template<typename T, int M, int N> qMatrix<T, 1, N> ddot_dx_const(const qMatrix<T, M, N>& da_dx, const qMatrix<T, M, 1>& b)
{
	return ddot_dx(qMatrix<T, M, 1>::Zero(), da_dx, b, qMatrix<T, M, N>::Zero());
}

template<typename T, int M, int N> qMatrix<T, 1, N> dnorm2_dx(const qMatrix<T, M, 1>& v, const qMatrix<T, M, N>& dv_dx)
{
	return 0.5f * v.transpose() * v * ddot_dx(v, dv_dx);
}

qMatrix<float, 3, 4> A(const Frame& i, const Frame& j)
{
	qMatrix<float, 3, 4> a;
	a.col(0, Q(-i.s));
	a.col(1, Q(-i.t));
	a.col(2, Q(j.s));
	a.col(3, Q(j.t));
	return a;
}

qMatrix<float, 1, 6> dfi_diffuse_du123_v123(const Vec3f& x_prev, const Vec3f& x_i, const Vec3f& x_next,
	const Frame& f_prev, const Frame& f_i, const Frame& f_next, float tau)
{
	float d = distance(x_next, x_i), d2 = d * d;
	qMatrix<float, 3, 4> a = A(f_i, f_next);
	qMatrix<float, 3, 4> p1 = (-1.0f / d2) * Q(x_next - x_i) * ddot_dx(Q(x_next - x_i), a);
	qMatrix<float, 3, 4> p2 = 1.0f / d * a;
	qMatrix<float, 1, 4> dot = ddot_dx_const(p1 + p2, Q(f_i.n));
	qMatrix<float, 1, 6> C = qMatrix<float, 1, 6>::Zero();
	C.submat<0, 2, 0, 5>(dot);
	return tau / PI * C;
	/*float coeff = tau / (PI * distanceSquared(x_next, x_i));
	qMatrix<float, 3, 1> n = Q(f_i.n), v = Q(x_next - x_i);
	qMatrix<float, 3, 3> H = (qMatrix<float, 3, 3>::Id() - v * v.transpose());
	qMatrix<float, 3, 6> C;
	C.zero();
	C.submat<0, 2, 2, 5>(A(f_i, f_next));
	return coeff * n.transpose() * H * C;*/
}

qMatrix<float, 1, 4> dG_du12_v12(const Vec3f& x_i, const Vec3f& x_j, const Frame& f_i, const Frame& f_j)
{
	qMatrix<float, 3, 1> n_i = Q(f_i.n), n_j = Q(f_j.n);
	float a = dot(x_j - x_i, f_i.n), b = dot(x_i - x_j, f_j.n), c = math::sqr(distanceSquared(x_j, x_i));

	qMatrix<float, 1, 4> da_du12_v12 = ddot_dx_const(A(f_i, f_j), Q(f_i.n));
	qMatrix<float, 1, 4> db_du12_v12 = ddot_dx_const(-A(f_i, f_j), Q(f_j.n));
	qMatrix<float, 1, 4> dc_du12_v12 = 2 * c * ddot_dx(Q(x_j - x_i), A(f_i, f_j));
	return ((da_du12_v12 * b + a * db_du12_v12) * c - a * b * dc_du12_v12) / (c * c);

	//qMatrix<float, 1, 3> p1 = c * (b * n_i.transpose() - a * n_j.transpose()), p2 = a * b * 4 * math::sqrt(c) * Q(x_j -  x_i).transpose();
	//qMatrix<float, 1, 3> p3 = (p1 - p2) / math::sqr(c);
	//return p3 * A(f_i, f_j);
}

}