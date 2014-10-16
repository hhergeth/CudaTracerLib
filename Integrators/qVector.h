#include "qMatrix.h"

template<typename T, int M> CUDA_FUNC_IN T dot(const qMatrix<T, M, 1>& lhs, const qMatrix<T, M, 1>& rhs)
{
	return (lhs * rhs.Transpose())(0, 0);
}

template<typename T, int M> CUDA_FUNC_IN T norm_dot(const qMatrix<T, M, 1>& lhs, const qMatrix<T, M, 1>& rhs)
{
	return (lhs * rhs.Transpose()) / (lhs.p_norm(T(2)) * rhs.p_norm(T(2)));
}

template<typename T, int M, int N> CUDA_FUNC_IN qMatrix<T, M, N> outer(const qMatrix<T, M, 1>& lhs, const qMatrix<T, N, 1>& rhs)
{
	qMatrix<T, M, N> r;
	for(int i = 0; i < M; i++)
		for(int j = 0; j < N; j++)
		{
			r(i, j) = lhs(i, 0) * rhs(j, 0);
		}
	return r;
}

template<typename T, int M> CUDA_FUNC_IN T norm2(const qMatrix<T, M, 1>& v)
{
	return v.p_norm(T(2));
}

template<typename T, int M> CUDA_FUNC_IN qMatrix<T, M, 1> normalize(const qMatrix<T, M, 1>& v, const T& p)
{
	return v / v.p_norm(p);
}

template<typename T, int M> CUDA_FUNC_IN qMatrix<T, M, 1> normalize(const qMatrix<T, M, 1>& v)
{
	return normalize(v, T(2));
}
