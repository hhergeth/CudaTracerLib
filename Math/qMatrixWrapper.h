#pragma once
#include <qMatrixLib/qMatrix.h>
#include <qMatrixLib/qMatrixHelper.h>
#include "Vector.h"
#include "float4x4.h"

namespace CudaTracerLib {

template<typename T, int N, typename INTERNAL> qMatrix<T, N, 1> Q(const VectorBase<T, N, INTERNAL>& v)
{
	qMatrix<T, N, 1> r;
	for (int i = 0; i < N; i++)
		r(i) = v[i];
	return r;
}

template<typename INTERNAL, typename T, int N> VectorBase<T, N, INTERNAL> Q(const qMatrix<T, N, 1>& v)
{
	VectorBase<T, N, INTERNAL> r;
	for (int i = 0; i < N; i++)
		r[i] = v(i);
	return r;
}

template<typename T> float4x4 Q(const qMatrix<T, 4, 4>& A)
{
	float4x4 r;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			r(i, j) = A(i, j);
	return r;
}

template<typename T> qMatrix<T, 4, 4> Q(const float4x4& A)
{
	qMatrix<T, 4, 4> r;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			r(i, j) = A(i, j);
	return r;
}

}