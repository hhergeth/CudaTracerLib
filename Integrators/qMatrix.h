#pragma once

#include <iostream>     // std::cout
#include <sstream>      // std::istringstream
#include <string>       // std::string
#include <iomanip>
#include <cmath>

#ifndef CUDA_FUNC_IN
#define CUDA_FUNC_IN inline
#endif

#define MATRIX_ELEMENT_FUNC(FUNC_HEADER, FUNC) \
	CUDA_FUNC_IN qMatrix<T, M, N> FUNC_HEADER const \
	{ \
		qMatrix<T, M, N> res; \
		for(int i = 0; i < M; i++) \
			for(int j = 0; j < N; j++) \
				res(i, j) = FUNC; \
		return res; \
	}

#define MATRIX_ELEMENT_FUNC_2(NAME, FUNC) \
	MATRIX_ELEMENT_FUNC(NAME(), FUNC(operator()(i, j)))

#define MATRIX_ELEMENT_FUNC_3(NAME) \
	MATRIX_ELEMENT_FUNC_2(NAME, std::NAME)

#ifndef DMIN2
#define DMIN2(A, B) ((A) < (B) ? (A) : (B))
#endif

#ifndef DMAX2
#define DMAX2(A, B) ((A) > (B) ? (A) : (B))
#endif

template <typename T> CUDA_FUNC_IN int qMatrix_sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}

template <typename T> CUDA_FUNC_IN T qMatrix_round( T val )
{
	if( val < 0 ) return std::ceil(val - 0.5);
	return std::floor(val + 0.5);
}

template<typename T> CUDA_FUNC_IN void qMatrix_swap(T& a, T& b)
{
	T tmp = a;
	a = b;
	b = tmp;
}

template<typename T> CUDA_FUNC_IN T qMatrix_sqr(T val)
{
	return val * val;
}

enum qDirection { qColumn, qRow };

enum qOperation { qMulLeft, qMulRight, qAdd, qSub};

template<typename T, int M, int N> struct qMatrix
{
	T dat[DMAX2(M * N, 1)];

	CUDA_FUNC_IN static qMatrix<T, M, N> Zero()
	{
		qMatrix<T, M, N> r;
		for(int i = 0; i < M; i++)
			for(int j = 0; j < N; j++)
				r(i, j) = 0;
		return r;
	}

	CUDA_FUNC_IN static qMatrix<T, M, N> Id()
	{
		qMatrix<T, M, N> r;
		r.id();
		return r;
	}

	CUDA_FUNC_IN static qMatrix<T, M, N> Ones()
	{
		qMatrix<T, M, N> r;
		r.ones();
		return r;
	}

	CUDA_FUNC_IN static qMatrix<T, M, 1> e(int i)
	{
		qMatrix<T, M, 1> r = qMatrix<T, M, 1>::Zero();
		r(i, 0) = 1.0f;
		return r;
	}

	CUDA_FUNC_IN static qMatrix<T, M, N> Parse(const char* s)
	{
		std::istringstream iss(s);
		qMatrix<T, M, N> r;
		for(int i = 0; i < M * N; i++)
		{
			float f;
			iss >> f;
			r.dat[i] = f;
		}
		return r;
	}

	CUDA_FUNC_IN static qMatrix<T, M, N> Vandermonde(const qMatrix<T, N, 1>& alpha)
	{
		qMatrix<T, M, N> r;
		for(int i = 0; i < M; i++)
		{
			T val = 1;
			for(int j = 0; j < N; j++)
			{
				r(i, j) = val;
				val *= alpha(i, 0);
			}
		}
		return r;
	}

	CUDA_FUNC_IN void id()
	{
		zero();
		for(int i = 0; i < DMIN2(M, N); i++)
			operator()(i, i) = 1;
	}

	CUDA_FUNC_IN void zero()
	{
		fill(T(0));
	}

	CUDA_FUNC_IN void ones()
	{
		fill(T(1));
	}

	CUDA_FUNC_IN void fill(const T& val)
	{
		for(int i = 0; i < M; i++)
			for(int j = 0; j < N; j++)
				operator()(i, j) = val;
	}

	CUDA_FUNC_IN const T& operator()(int i, int j) const
	{
#ifndef ISCUDA
		if(i >= M || j >= N)
			throw std::runtime_error("Invalid matrix access.");
#endif
		return dat[i * N + j];
	}

	CUDA_FUNC_IN T& operator()(int i, int j)
	{
#ifndef ISCUDA
		if(i >= M || j >= N)
			throw std::runtime_error("Invalid matrix access.");
#endif
		return dat[i * N + j];
	}

	CUDA_FUNC_IN const T& operator()(int i) const
	{
		if(is_colvec())
			return operator()(i, 0);
		else if(is_rowvec())
			return operator()(0, i);
#ifndef __CUDAARCH__
		else throw std::runtime_error("Invalid matrix access.");
#endif 
	}

	CUDA_FUNC_IN T& operator()(int i)
	{
		if(N == 1)//col
			return operator()(i, 0);
		else if(M == 1)//row
			return operator()(0, i);
#ifndef __CUDAARCH__
		else throw std::runtime_error("Invalid matrix access.");
#endif 
	}

	CUDA_FUNC_IN bool operator==(const qMatrix<T, M, N>& rhs) const
	{
		for(int i = 0; i < M; i++)
			for(int j = 0; j < N; j++)
				if(operator()(i, j) != rhs(i, j))
					return false;
		return true;
	}

	CUDA_FUNC_IN bool operator!=(const qMatrix<T, M, N>& rhs) const
	{
		for(int i = 0; i < M; i++)
			for(int j = 0; j < N; j++)
				if(operator()(i, j) != rhs(i, j))
					return true;
		return false;
	}

	CUDA_FUNC_IN size_t n_rows() const
	{
		return M;
	}

	CUDA_FUNC_IN size_t n_cols() const
	{
		return N;
	}

	//first_row, first_col, last_row, last_col
	template<int p, int r, int q, int s> CUDA_FUNC_IN qMatrix<T, q - p + 1, s - r + 1> submat() const
	{
		qMatrix<T, q - p + 1, s - r + 1> res;
		for(int i = p; i <= q; i++)
			for(int j = r; j <= s; j++)
				res(i - p, j - r) = operator()(i, j);
		return res;
	}

	template<int p, int r, int q, int s> CUDA_FUNC_IN void submat(const qMatrix<T, q - p + 1, s - r + 1>& sub)
	{
		for(int i = p; i <= q; i++)
			for(int j = r; j <= s; j++)
				operator()(i, j) = sub(i - p, j - r);
	}

	//selects columns ie p < M, q < M, p <= q
	template<int p, int q> CUDA_FUNC_IN qMatrix<T, N, q - p + 1> cols() const
	{
		return submat<0, p, M, q>();
	}

	template<int p, int q> CUDA_FUNC_IN void cols(const qMatrix<T, N, q - p + 1>& cols)
	{
		submat<0, p, M, q>(cols);
	}

	//selects rows ie r < N, s < N, r <= s
	template<int r, int s> CUDA_FUNC_IN qMatrix<T, s - r + 1, M> rows() const
	{
		return submat<r, 0, s, N>();
	}

	template<int r, int s> CUDA_FUNC_IN  void rows(const qMatrix<T, s - r + 1, M>& rows)
	{
		submat<r, 0, s, N>(rows);
	}

	CUDA_FUNC_IN qMatrix<T, 1, N> row(int j) const
	{
		qMatrix<T, 1, N> r;
		for(int k = 0; k < N; k++)
			r(0, k) = operator()(j, k);
		return r;
	}

	CUDA_FUNC_IN qMatrix<T, M, 1> col(int i) const
	{
		qMatrix<T, M, 1> r;
		for(int k = 0; k < M; k++)
			r(k, 0) = operator()(k, i);
		return r;
	}

	CUDA_FUNC_IN void row(int i, const qMatrix<T, 1, N>& r)
	{
		for(int j = 0; j < N; j++)
			operator()(i, j) = r(0, j);
	}

	CUDA_FUNC_IN void col(int j, const qMatrix<T, M, 1>& c)
	{
		for(int i = 0; i < M; i++)
			operator()(i, j) = c(i, 0);
	}

	CUDA_FUNC_IN qMatrix<T, DMIN2(M, N), 1> diag() const
	{
		qMatrix<T, DMIN2(M, N), 1> res;
		for(int i = 0; i < DMIN2(M, N); i++)
			res(i, 0) = operator()(i, i);
		return res;
	}

	CUDA_FUNC_IN void diag(const qMatrix<T, DMIN2(M, N), 1>& d)
	{
		for(int i = 0; i < DMIN2(M, N); i++)
			operator()(i, i) = d(i, 0);
	}

	CUDA_FUNC_IN void swap_rows(int r, int s)
	{
		for(int j = 0; j < N; j++)
			qMatrix_swap(operator()(r, j), operator()(s, j));
	}

	CUDA_FUNC_IN void swap_cols(int p, int q)
	{
		for(int i = 0; i < M; i++)
			qMatrix_swap(operator()(i, p), operator()(i, q));
	}

	CUDA_FUNC_IN qMatrix<T, N, M> fliplr() const
	{
		qMatrix<T, N, M> res;
		for(int j = 0; j < N; j++)
			res.col(N - j - 1, col(j));
		return res;
	}

	CUDA_FUNC_IN qMatrix<T, N, M> flipud() const
	{
		qMatrix<T, N, M> res;
		for(int i = 0; i < M; i++)
			res.row(M - i - 1, row(i));
		return res;
	}

	CUDA_FUNC_IN qMatrix<T, N, M> Transpose() const
	{
		qMatrix<T, N, M> r;
		for(int i = 0; i < M; i++)
			for(int j = 0; j < N; j++)
				r(j, i) = operator()(i, j);
		return r;
	}

	template<int R> CUDA_FUNC_IN qMatrix<T, M, N + R> JoinHorizontal(const qMatrix<T, M, R>& rhs) const
	{
		qMatrix<T, M, N + R> res;
		for(int i = 0; i < M; i++)
			for(int j = 0; j < N + R; j++)
				res(i, j) = j < N ? operator()(i, j) : rhs(i, j - N);
		return res;
	}

	template<int R> CUDA_FUNC_IN qMatrix<T, M + R, N> JoinVertical(const qMatrix<T, R, N>& rhs) const
	{
		qMatrix<T, M + R, N> res;
		for(int i = 0; i < M + R; i++)
			for(int j = 0; j < N; j++)
				res(i, j) = i < M ? operator()(i, j) : rhs(i - M, j);
		return res;
	}

	CUDA_FUNC_IN qMatrix<T, M, N> Add(const qMatrix<T, M, N>& rhs) const
	{
		qMatrix<T, M, N> r;
		for(int i = 0; i < M; i++)
			for(int j = 0; j < N; j++)
				r(i, j) = operator()(i, j) + rhs(i, j);
		return r;
	}

	CUDA_FUNC_IN qMatrix<T, M, N> Add(const T& val) const
	{
		qMatrix<T, M, N> r;
		for(int i = 0; i < M; i++)
			for(int j = 0; j < N; j++)
				r(i, j) = operator()(i, j) + val;
		return r;
	}

	CUDA_FUNC_IN qMatrix<T, M, N> Sub(const qMatrix<T, M, N>& rhs) const
	{
		qMatrix<T, M, N> r;
		for(int i = 0; i < M; i++)
			for(int j = 0; j < N; j++)
				r(i, j) = operator()(i, j) - rhs(i, j);
		return r;
	}

	CUDA_FUNC_IN qMatrix<T, M, N> Sub(const T& val) const
	{
		qMatrix<T, M, N> r;
		for(int i = 0; i < M; i++)
			for(int j = 0; j < N; j++)
				r(i, j) = operator()(i, j) - val;
		return r;
	}

	CUDA_FUNC_IN qMatrix<T, M, N> Mul(const T& val) const
	{
		qMatrix<T, M, N> r;
		for(int i = 0; i < M; i++)
			for(int j = 0; j < N; j++)
				r(i, j) = operator()(i, j) * val;
		return r;
	}

	CUDA_FUNC_IN qMatrix<T, M, N> Div(const T& val) const
	{
		qMatrix<T, M, N> r;
		for(int i = 0; i < M; i++)
			for(int j = 0; j < N; j++)
				r(i, j) = operator()(i, j) / val;
		return r;
	}

	CUDA_FUNC_IN qMatrix<T, M, N> MulElement(const qMatrix<T, M, N>& rhs) const
	{
		qMatrix<T, M, N> r;
		for(int i = 0; i < M; i++)
			for(int j = 0; j < N; j++)
				r(i, j) = operator()(i, j) * rhs(i, j);
		return r;
	}

	CUDA_FUNC_IN qMatrix<T, M, N> DivElement(const qMatrix<T, M, N>& rhs) const
	{
		qMatrix<T, M, N> r;
		for(int i = 0; i < M; i++)
			for(int j = 0; j < N; j++)
				r(i, j) = operator()(i, j) / rhs(i, j);
		return r;
	}

	template<int R> CUDA_FUNC_IN qMatrix<T, M, R>  Mul(const qMatrix<T, N, R>& rhs) const
	{
		qMatrix<T, M, R> r;
		for(int i = 0; i < M; i++)
			for(int j = 0; j < R; j++)
			{
				T val = 0;
				for(int k = 0; k < N; k++)
					val += operator()(i, k) * rhs(k, j);
				r(i, j) = val;
			}
		return r;
	}

	CUDA_FUNC_IN qMatrix<T, M, N> operator++() const
	{
		return this->Add(T(1));
	}

	CUDA_FUNC_IN qMatrix<T, M, N> operator--() const
	{
		return this->Sub(T(1));
	}

	CUDA_FUNC_IN qMatrix<T, M, N> operator-() const 
	{
		return this->Mul(T(-1));
	}

	MATRIX_ELEMENT_FUNC(clamp(const T& low, const T& high), operator()(i, j) < low ? low : (operator()(i, j) > high ? high : operator()(i, j)))

	MATRIX_ELEMENT_FUNC_3(abs)

	MATRIX_ELEMENT_FUNC_3(exp)

	MATRIX_ELEMENT_FUNC_3(log)

	MATRIX_ELEMENT_FUNC_3(sqrt)

	MATRIX_ELEMENT_FUNC(pow(const T& p), std::pow(operator()(i, j), p))

	MATRIX_ELEMENT_FUNC(sqr(), operator()(i, j) * operator()(i, j))

	MATRIX_ELEMENT_FUNC_3(floor)

	MATRIX_ELEMENT_FUNC_3(ceil)

	MATRIX_ELEMENT_FUNC_2(sign, signf)

	MATRIX_ELEMENT_FUNC_3(cos)
	MATRIX_ELEMENT_FUNC_3(acos)
	MATRIX_ELEMENT_FUNC_3(cosh)
	//MATRIX_ELEMENT_FUNC_3(acosh)

	MATRIX_ELEMENT_FUNC_3(sin)
	MATRIX_ELEMENT_FUNC_3(asin)
	MATRIX_ELEMENT_FUNC_3(sinh)
	//MATRIX_ELEMENT_FUNC_3(asinh)

	MATRIX_ELEMENT_FUNC_3(tan)
	MATRIX_ELEMENT_FUNC_3(atan)
	MATRIX_ELEMENT_FUNC_3(tanh)
	//MATRIX_ELEMENT_FUNC_3(atanh)

	CUDA_FUNC_IN T accu() const
	{
		T res = T();
		for(int i = 0; i < M; i++)
			for(int j = 0; j < N; j++)
				res += operator()(i, j);
		return res;
	}

	CUDA_FUNC_IN T accuabs() const
	{
		T res = T();
		for(int i = 0; i < M; i++)
			for(int j = 0; j < N; j++)
				res += std::abs(operator()(i, j));
		return res;
	}

	CUDA_FUNC_IN T accuabs(int row_start, int col_start, int row_end, int col_end) const
	{
		T res = T();
		for(int i = row_start; i <= row_end; i++)
			for(int j = col_start; j <= col_end; j++)
				res += std::abs(operator()(i, j));
		return res;
	}
	
	CUDA_FUNC_IN T min() const
	{
		T res = std::numeric_limits<T>::max();
		for(int i = 0; i < M; i++)
			for(int j = 0; j < N; j++)
				res = std::min(res, operator()(i, j));
		return res;
	}

	CUDA_FUNC_IN T max() const
	{
		T res = std::numeric_limits<T>::min();
		for(int i = 0; i < M; i++)
			for(int j = 0; j < N; j++)
				res = std::max(res, operator()(i, j));
		return res;
	}

	CUDA_FUNC_IN T mean() const
	{
		return accu() / T(M * N);
	}

	CUDA_FUNC_IN T var() const
	{
		T res = T(), m = mean();
		for(int i = 0; i < M; i++)
			for(int j = 0; j < N; j++)
			{
				T f = operator()(i, j) - m;
				res += f * f;
			}
		return res / T(M * N);
	}

	CUDA_FUNC_IN T stddev() const
	{
		return std::sqrt(var());
	}

	CUDA_FUNC_IN T p_norm(T p) const
	{
		T r = T();
		for(int i = 0; i < M; i++)
			for(int j = 0; j < N; j++)
				r += ::pow(operator()(i, j), p);
		return ::pow(r, T(1.0) / p);
	}

	CUDA_FUNC_IN T max_norm() const
	{
		T r = T();
		for(int i = 0; i < M; i++)
			for(int j = 0; j < N; j++)
				r = MAX(r, std::abs(operator()(i, j)));
		return r;
	}

	CUDA_FUNC_IN T col_sum_norm() const
	{
		T r = T();
		for(int j = 0; j < N; j++)
		{
			T s = T();
			for(int i = 0; i < M; i++)
				s += std::abs(operator()(i, j));
			r = MAX(r, s);
		}
		return r;
	}

	CUDA_FUNC_IN T row_sum_norm() const
	{
		T r = T();
		for(int i = 0; i < M; i++)
		{
			T s = T();
			for(int j = 0; j < N; j++)
				s += std::abs(operator()(i, j));
			r = MAX(r, s);
		}
		return r;
	}

	CUDA_FUNC_IN qMatrix<T, M, N> Round(int digts_after_decimal) const
	{
		qMatrix<T, M, N> res;
		T f = std::pow((T)10, (T)(digts_after_decimal));
		for(int i = 0; i < M; i++)
			for(int j = 0; j < N; j++)
				res(i, j) = qMatrix_round( operator()(i, j) * f ) / f;
		return res;
	}

	CUDA_FUNC_IN bool is_finite() const
	{
		for(int i = 0; i < M; i++)
			for(int j = 0; j < N; j++)
				if(operator()(i, j) != operator()(i, j))// || std::isfinite(operator()(i, j))
					return false;
		return true;
	}

	CUDA_FUNC_IN bool is_vec() const
	{
		return is_colvec() || is_rowvec();
	}

	CUDA_FUNC_IN bool is_colvec() const
	{
		return N == 1;
	}

	CUDA_FUNC_IN bool is_rowvec() const
	{
		return M == 1;
	}

	CUDA_FUNC_IN bool is_quadratic() const
	{
		return M == N;
	}

	CUDA_FUNC_IN bool is_symmetric() const
	{
		if(!is_quadratic())
			return false;
		for(int i = 0; i < M; i++)
			for(int j = i + 1; j < N; j++)
				if(operator()(i, j) != operator()(j, i))
					return false;
		return true;
	}

	template<typename U> CUDA_FUNC_IN qMatrix<U, M, N> convert()
	{
		qMatrix<U, M, N> res;
		for(int i = 0; i < M; i++)
			for(int j = 0; j < N; j++)
				res(i, j) = U(operator()(i, j));
		return res;
	}

	template<typename U> operator qMatrix<U, M, N>()
	{
		return convert<U>();
	}

	void print(std::ostream &os) const
	{
		std::ostringstream str;
		size_t w = 0;
		for(int i = 0; i < M; ++i)
			for(int j = 0; j < N; ++j)
			{
				str << operator()(i, j);
				w = DMAX2((size_t)str.tellp(), w);
				str.str("");
			}

		for(int i = 0; i < M; ++i)
		{
			for(int j = 0; j < N; ++j)
			{
				os << std::right << std::setw(w + 1) << operator()(i, j);
			}

			os << std::endl;
		}
	}

	friend std::ostream & operator<<(std::ostream &os, const qMatrix<T, M, N>& p)
	{
		p.Round(3).print(os);
		return os;
	}
};

template<typename T, int N> CUDA_FUNC_IN qMatrix<T, N, N> diagmat(const qMatrix<T, N, 1>& diag)
{
	qMatrix<T, N, N> res;
	res.zero();
	res.diag(diag);
	return res;
}

template<typename T, int M, int N> CUDA_FUNC_IN qMatrix<T, M, N> operator+(qMatrix<T, M, N> const& lhs, qMatrix<T, M, N> const& rhs) 
{
	return lhs.Add(rhs);
}

template<typename T, int M, int N> CUDA_FUNC_IN qMatrix<T, M, N> operator-(qMatrix<T, M, N> const& lhs, qMatrix<T, M, N> const& rhs) 
{
	return lhs.Sub(rhs);
}

template<typename T, int M, int N> CUDA_FUNC_IN qMatrix<T, M, N> operator+(qMatrix<T, M, N> const& lhs, T const& rhs) 
{
	return lhs.Add(rhs);
}

template<typename T, int M, int N> CUDA_FUNC_IN qMatrix<T, M, N> operator-(qMatrix<T, M, N> const& lhs, T const& rhs) 
{
	return lhs.Sub(rhs);
}

template<typename T, int M, int N> CUDA_FUNC_IN qMatrix<T, M, N> operator*(qMatrix<T, M, N> const& lhs, const T& rhs) 
{
	return lhs.Mul(rhs);
}

template<typename T, int M, int N> CUDA_FUNC_IN qMatrix<T, M, N> operator/(qMatrix<T, M, N> const& lhs, const T& rhs) 
{
	return lhs.Div(rhs);
}

template<typename T, int M, int N> CUDA_FUNC_IN qMatrix<T, M, N> operator%(qMatrix<T, M, N> const& lhs, qMatrix<T, M, N> const& rhs) 
{
	return lhs.Mul(rhs);
}

template<typename T, int M, int N> CUDA_FUNC_IN qMatrix<T, M, N> operator/(qMatrix<T, M, N> const& lhs, qMatrix<T, M, N> const& rhs) 
{
	return lhs.Div(rhs);
}

template<typename T, int M, int N> CUDA_FUNC_IN qMatrix<T, M, N> operator*(const T& lhs, qMatrix<T, M, N> const& rhs) 
{
	return rhs.Mul(lhs);
}

template<typename T, int M, int N, int R> CUDA_FUNC_IN qMatrix<T, M, R> operator*(qMatrix<T, M, N> const& lhs, qMatrix<T, N, R> const& rhs) 
{
	return lhs.Mul(rhs);
}

template<typename T, int N> CUDA_FUNC_IN qMatrix<T, N, 1> linspace(const T& start, const T& end, const T& n)
{
	T f = (end - start) / n;
	qMatrix<T, N, 1> res;
	for(int i = 0; i < N; i++)
		res(i, 0) = start + f * i;
	return res;
}

template<typename T, int N> CUDA_FUNC_IN qMatrix<T, N, 1> linspace(const T& start, const T& end)
{
	return linspace(start, end, end - start);
}

template<typename T, int M, int N, int P, int R> CUDA_FUNC_IN qMatrix<T, M * P, N * R> kron(const qMatrix<T, M, N>& lhs, const qMatrix<T, P, R>& rhs)
{
	qMatrix<T, M * P, N * R> res;
	for(int i = 0; i < M; i++)
		for(int j = 0; j < N; j++)
			res.submat<P * i, R * j, P * (i + 1), R * (j + 1)>(lhs(i, j) * rhs);
	return res;
}

template<typename T, int N> CUDA_FUNC_IN qMatrix<T, N, N> symmatu(const qMatrix<T, N, N>& A)
{
	qMatrix<T, N, N> res = A;
	for(int i = 1; i < M; i++)
		for(int j = 0; j < i; j++)
			res(i, j) = A(j, i);
	return res;
}

template<typename T, int N> CUDA_FUNC_IN qMatrix<T, N, N> symmatl(const qMatrix<T, N, N>& A)
{
	qMatrix<T, N, N> res = A;
	for(int j = 1; j < N; j++)
		for(int i = 0; i < j; i++)
			res(i, j) = A(j, i);
	return res;
}

template<typename T> CUDA_FUNC_IN void eig2x2(const qMatrix<T, 2, 2>& A, T& l1, T& l2)
{
	int N = 2;
	T a = A(N - 2,  N - 2), b = A(N - 2, N - 1), c = A(N - 1, N - 2), d = A(N - 1, N - 1), p = -a - d, q = a * d - b * c;
	T l0 = std::sqrt(p * p / T(4) - q);
	l1 = -p / T(2) + l0;
	l2 = -p / T(2) - l0;
}

//interpret square matrix A as upper triangular
template<typename T, int N> CUDA_FUNC_IN qMatrix<T, N, N> trimatu(const qMatrix<T, N, N>& A)
{
	qMatrix<T, N, N> res = A;
	for(int i = 1; i < M; i++)
		for(int j = 0; j < i; j++)
			res(i, j) = T();
	return res;
}

//interpret square matrix A as lower triangular
template<typename T, int N> CUDA_FUNC_IN qMatrix<T, N, N> trimatl(const qMatrix<T, N, N>& A)
{
	qMatrix<T, N, N> res = A;
	for(int j = 1; j < N; j++)
		for(int i = 0; i < j; i++)
			res(i, j) = T();
	return res;
}

template<typename T, int N> CUDA_FUNC_IN T trace(const qMatrix<T, N, N>& A)
{
	return A.diag().accu();
}

template<typename T, int M, int N> CUDA_FUNC_IN qMatrix<T, M, N> minimize(const qMatrix<T, M, N>& A, const qMatrix<T, M, N>& B)
{
	qMatrix<T, M, N> res;
	for(int i = 0; i < M; i++)
		for(int j = 0; j < N; j++)
			res(i, j) = std::min(A(i, j), B(i, j));
	return res;
}

template<typename T, int M, int N> CUDA_FUNC_IN qMatrix<T, M, N> maximize(const qMatrix<T, M, N>& A, const qMatrix<T, M, N>& B)
{
	qMatrix<T, M, N> res;
	for(int i = 0; i < M; i++)
		for(int j = 0; j < N; j++)
			res(i, j) = std::max(A(i, j), B(i, j));
	return res;
}

template<typename T, int M, int N> struct MAT
{
	int i;
	qMatrix<T, M, N> m;
	CUDA_FUNC_IN MAT()
		: i(0)
	{
		m.zero();
	}
	CUDA_FUNC_IN MAT& operator%(const T& val)
	{
		m(i / N, i % N) = val;
		i++;
		return *this;
	}
	CUDA_FUNC_IN MAT& operator()()
	{
		i += M - (i % M);
		return *this;
	}
	CUDA_FUNC_IN operator qMatrix<T, M, N>() const
	{
		return m;
	}
};

template<typename T, int N> struct VEC : public MAT<T, N, 1>
{
};

#define MAKE_TYPEDEF(L, T) \
	typedef MAT<T, L, 1> q##T##L;  \
	typedef MAT<T, L, L> q##T##L##x##L; 

#define MAKE_ALL_TYPEDEFS(L) \
	MAKE_TYPEDEF(L, int) \
	MAKE_TYPEDEF(L, __int64) \
	MAKE_TYPEDEF(L, float) \
	MAKE_TYPEDEF(L, double)

MAKE_ALL_TYPEDEFS(1)
MAKE_ALL_TYPEDEFS(2)
MAKE_ALL_TYPEDEFS(3)
MAKE_ALL_TYPEDEFS(4)
MAKE_ALL_TYPEDEFS(5)

//decompositions

template<typename T, int M> CUDA_FUNC_IN qMatrix<T, M, 1> householder(const qMatrix<T, M, 1>& a)
{
	qMatrix<T, M, 1> e = qMatrix<T, M, 1>::e(0), u, v;
	T alpha = qMatrix_sgn(a(0, 0)) * a.p_norm(T(2));
	u = a - alpha * e;
	v = u / u.p_norm(T(2));
	return v;
}

template<typename T, int M, int N> CUDA_FUNC_IN qMatrix<T, M, 1> householder(const qMatrix<T, M, N>& A, int k)
{
	qMatrix<T, M, 1> a = A.col(k), e = qMatrix<T, M, 1>::e(k), u, v;
	if(a.accu() == a(0, 0))
		return a / a(0, 0);
	for(int i = 0; i < k; i++)
		a(i, 0) = 0;
	T alpha = qMatrix_sgn(a(k, 0)) * a.p_norm(T(2));
	u = a - alpha * e;
	v = u / u.p_norm(T(2));
	return v;
}

template<typename T, int M, int N> CUDA_FUNC_IN void qrHousholder(const qMatrix<T, M, N>& A, qMatrix<T, M, M>& Q, qMatrix<T, M, N>& R)
{
	Q.id();
	R = A;
	int K = DMIN2(M - 1, N);
	for(int k = 0; k < K; k++)
	{
		qMatrix<T, M, 1> v = householder(R, k);
		qMatrix<T, M, M> Q_k = qMatrix<T, M, M>::Id() - T(2.0) * v * v.Transpose();
		Q = Q * Q_k.Transpose();
		R = Q_k * R;
	}
}

template<typename T, int M, int N> CUDA_FUNC_IN void qrGivens(const qMatrix<T, M, N>& A, qMatrix<T, M, M>& Q, qMatrix<T, M, N>& R)
{
	Q.id();
	R = A;
	for(int j = 0; j < DMIN2(N - 1, M); j++)
		for(int i = j + 1; i < M; i++)
		{
			T a = R(j, j),b = R(i, j), r, c, s;
			if(std::abs(b) < T(1e-5))
				continue;
			if(b == 0)
			{
				c = (T)_copysign(T(1), a);
				s = 0;
				r = std::abs(a);
			}
			else if(a == 0)
			{
				c = 0;
				s = -(T)_copysign(T(1), b);
				r = std::abs(b);
			}
			else if(std::abs(b) > std::abs(a))
			{
				T t = a / b, u = (T)_copysign(std::sqrt(T(1) + t * t), b);
				s = -T(1) / u;
				c = -s * t;
				r = b * u;
			}
			else 
			{
				T t = b / a, u = (T)_copysign(std::sqrt(T(1) + t * t), a);
				c = T(1) / u;
				s = -c * t;
				r = a * u;
			}

			qMatrix<T, M, M> Q_k = qMatrix<T, M, M>::Id();
			Q_k(i, i) = Q_k(j, j) = c;
			Q_k(j, i) = -s;
			Q_k(i, j) = s;
			Q = Q * Q_k.Transpose();
			R = Q_k * R;
		}
}

template<typename T, int M, int N> CUDA_FUNC_IN void qrHousholderRR(const qMatrix<T, M, N>& A, qMatrix<T, M, M>& Q, qMatrix<T, M, N>& R, qMatrix<T, M, M>& P)
{
	P.id();
	qMatrix<T, N, 1> colNorms;
	for(int i = 0; i < N; i++)
		colNorms(i) = qMatrix_sqr(A.col(i).p_norm(T(2)));
	Q.id();
	R = A;
	int K = DMIN2(M - 1, N);
	for(int j = 0; j < K; j++)
	{
		int p = j;
		for(int i = j; i < K; i++)
			if(colNorms(i, 0) > colNorms(p, 0))
				p = i;
		if(colNorms(p) == 0)
			break;
		if(j != p)
		{
			P.swap_cols(p, j);
			R.swap_cols(p, j);
			colNorms.swap_rows(p, j);
		}
		qMatrix<T, M, 1> v = householder(R, p);
		R = R - v * (v.Transpose() * R);
		Q = Q - (Q * v) * v.Transpose();
		for(int i = j + 1; i < K; i++)
			colNorms(i, 0) = colNorms(i, 0) - qMatrix_sqr(R(j, i));
	}
	R = R * P.Transpose();
}


template<typename T, int N> CUDA_FUNC_IN void luDecomposition(const qMatrix<T, N, N>& A, qMatrix<T, N, N>& P, qMatrix<T, N, N>& L, qMatrix<T, N, N>& U)
{
	qMatrix<T, N, N> LR = A;
	int p[N];
	for(int i = 0; i < N; i++)
		p[i] = i;
	for(int j = 0; j < N - 1; j++)
	{
		int i_p = j;
		for(int i = 0; i < N; i++)
			if(std::abs(LR(i, j)) > std::abs(LR(i_p, j)))
				i_p = j;
		qMatrix_swap(p[i_p], p[j]);
		LR.swap_rows(j, i_p);
		for(int i = j + 1; i < N; i++)
		{
			LR(i, j) = LR(i, j) / LR(j, j);
			for(int k = j + 1; k < N; k++)
				LR(i, k) = LR(i, k) - LR(i, j) * LR(j, k);
		}
	}
	L = U = LR;
	for(int i = 0; i < N; i++)
		for(int j = 0; j < N; j++)
		{
			if(i == j)
				L(i, j) = 1;
			if(i < j)
				L(i, j) = 0;
			if(i > j)
				U(i, j) = 0;
		}
	P.zero();
	for(int i = 0; i < N; i++)
		P(i, p[i]) = 1;
}

namespace __hessenbergReduction__
{
	template<typename T, int N, int i> struct loop
	{
		static void exec(qMatrix<T, N, N>& A, qMatrix<T, N, N>& Q)
		{
			if(i < N - 2)
			{
				qMatrix<T, N - i - 1, 1> u = householder(A.template submat<i + 1, i, N - 1, i>());
				qMatrix<T, N - i - 1, N - i - 1> P_i = qMatrix<T, N - i - 1, N - i - 1>::Id() - T(2) * u * u.Transpose();
				A.template submat<i + 1, i, N - 1, N - 1>(P_i * A.template submat<i + 1, i, N - 1, N - 1>());
				A.template submat<0, i + 1, N - 1, N - 1>(A.template submat<0, i + 1, N - 1, N - 1>() * P_i);
				Q.template submat<i + 1, i, N - 1, N - 1>(P_i * Q.template submat<i + 1, i, N - 1, N - 1>());
				loop<T, N, i + 1>::exec(A, Q);
			}
		}
	};
	template<typename T, int N> struct loop<T, N, N>
	{
		static void exec(qMatrix<T, N, N>& A, qMatrix<T, N, N>& Q)
		{

		}
	};
}
template<typename T, int N> CUDA_FUNC_IN void hessenbergReduction(const qMatrix<T, N, N>& A, qMatrix<T, N, N>& H, qMatrix<T, N, N>& Q)
{
	H = A;
	Q.id();
	__hessenbergReduction__::loop<T, N, 0>::exec(H, Q);
}

//X has to be symmetric and of full rank
template<typename T, int N> CUDA_FUNC_IN void qrAlgorithmSymmetric(const qMatrix<T, N, N>& X, qMatrix<T, N, N>& D, qMatrix<T, N, N>& V, int n = 50)
{
	//using Wilkinson shifts
	V.id();
	qMatrix<T, N, N> X_i = X, I = qMatrix<T, N, N>::Id();
	for(int i = 0; i < n; i++)
	{
		T kappa = 0;
		if(N > 2)
		{
			T l1, l2, d = X_i(N - 1, N - 1);
			eig2x2(X_i.template submat<N - 2, N - 2, N - 1, N - 1>(), l1, l2);
			kappa = std::abs(l1 - d) < std::abs(l2 - d) ? l1 : l2;
		}

		qMatrix<T, N, N> Q_i, R_i;
		qrHousholder(X_i - kappa * I, Q_i, R_i);
		X_i = R_i * Q_i + kappa * I;
		V = V * Q_i;
	}
	D = X_i;
}

template<typename T, int N> CUDA_FUNC_IN qMatrix<T, N, 1> inversePowerMethod(const qMatrix<T, N, N>& A, const T& lambda)
{
	qMatrix<T, N, 1> w = solve(A - lambda * qMatrix<T, N, N>::Id(), qMatrix<T, N, 1>::e(0));
	return w / w.p_norm(T(2));
}

template<typename T, int N> CUDA_FUNC_IN void qrAlgorithm(const qMatrix<T, N, N>& X, qMatrix<T, N, N>& D, qMatrix<T, N, N>& V, int n = 50)
{
	V.id();
	qMatrix<T, N, N> X_i = X;
	for(int i = 0; i < n; i++)
	{
		qMatrix<T, N, N> Q_i, R_i, P;
		qrHousholder(X_i, Q_i, R_i);
		X_i = R_i * Q_i;
		V = V * Q_i;
	}
	D = X_i;
	V.zero();
	int i = 0;
	while(i < N && std::abs(D(i, i)) > 1e-5)
	{
		V.col(i, inversePowerMethod(X, D(i, i)));
		i++;
	}
}
/*
template<typename T, int N> CUDA_FUNC_IN void qrAlgorithmDoubleShift(const qMatrix<T, N, N>& X, qMatrix<T, N, N>& D, qMatrix<T, N, N>& V, int n = 50)
{
	V.id();
	qMatrix<T, N, N> X_i = X, I = qMatrix<T, N, N>::Id();
	for(int i = 0; i < n; i++)
	{
		T kappa1 = X_i(N - 2, N - 2) + X_i(N - 1, N - 1), kappa2 = X_i(N - 2, N - 2) * X_i(N - 1, N - 1) - X_i(N - 2, N - 1) * X_i(N - 1, N - 2);
		qMatrix<T, N, N> p_A = X_i * X_i - kappa1 * X_i + kappa2 * I;
		qMatrix<T, N, N> Q_i, R_i;
		qrHousholder(p_A, Q_i, R_i);
		X_i = Q_i.Transpose() * X_i * Q_i;
		V = V * Q_i;
	}
	D = X_i;
}
*/
template<typename T, int M, int N> CUDA_FUNC_IN int svd(const qMatrix<T, M, N>& A, qMatrix<T, M, M>& U, qMatrix<T, N, N>& V, qMatrix<T, M, N>& S, T eps = T(-1))
{
	qMatrix<T, M, N> B = A;
	std::cout << "A : " << std::endl << A << std::endl;
	U = qMatrix<T, M, M>::Id();
	V = qMatrix<T, N, N>::Id();
	for(int k = 0; k < DMIN2(M - 1, N); k++)
	{
		qMatrix<T, M, 1> u = householder(B, k);
		qMatrix<T, M, M> U_k = qMatrix<T, M, M>::Id() - T(2) * u * u.Transpose();
		B = U_k.Transpose() * B;

		qMatrix<T, N, N> V_k = qMatrix<T, N, N>::Id();
		if(k < N - 2)
		{
			qMatrix<T, 1, N> b = B.row(k), v;
			for(int i = 0; i <= k; i++)
				b(0, i) = 0;
			T alpha = qMatrix_sgn(b(0, k + 1)) * b.p_norm(T(2));
			v = b - alpha * qMatrix<T, M, 1>::e(k + 1).Transpose();
			v = v / v.p_norm(T(2));
			V_k = qMatrix<T, N, N>::Id() - T(2) * v.Transpose() * v;
		}

		U = U_k * U;
		V = V * V_k;
		B = B * V_k;
	}

	qMatrix<T, M, M> ev0, ev1, U2, V2;
	qrAlgorithmSymmetric(B * B.Transpose(), ev0, U2);
	qrAlgorithmSymmetric(B.Transpose() * B, ev1, V2);

	S = diagmat(ev0.diag().sqrt());
	U = U.Transpose() * U2;
	V = V * V2;
	
	std::cout << "U : " << std::endl << U << std::endl;
	std::cout << "S : " << std::endl << S << std::endl;
	std::cout << "V : " << std::endl << V << std::endl;
	std::cout << "USV' : " << std::endl << (U * S * V.Transpose()) << std::endl;
	return 1;
}

template<typename T, int N> CUDA_FUNC_IN qMatrix<T, N, 1> solveUpperDiagonal(const qMatrix<T, N, N>& U, const qMatrix<T, N, 1>& rhs)
{
	qMatrix<T, N, 1> r;
	for(int i = N - 1; i >= 0; i--)
	{
		T val = 0;
		for(int j = i + 1; j < N; j++)
			val += U(i, j) * r(j, 0);
		r(i, 0) = (rhs(i, 0) - val) / U(i, i);
	}
	return r;
}

template<typename T, int N> CUDA_FUNC_IN qMatrix<T, N, 1> solveLowerDiagonal(const qMatrix<T, N, N>& L, const qMatrix<T, N, 1>& rhs)
{
	qMatrix<T, N, 1> r;
	for(int i = 0; i < N; i++)
	{
		T val = 0;
		for(int j = 0; j < i; j++)
			val += L(i, j) * r(j, 0);
		r(i, 0) = (rhs(i, 0) - val) / L(i, i);
	}
	return r;
}

template<typename T, int N> CUDA_FUNC_IN qMatrix<T, N, 1> solve(const qMatrix<T, N, N>& P, const qMatrix<T, N, N>& L, const qMatrix<T, N, N>& U, const qMatrix<T, N, 1>& rhs)
{
	qMatrix<T, N, 1> b = P * rhs;
	qMatrix<T, N, 1> d = solveLowerDiagonal(L, b);
	return solveUpperDiagonal(U, d);
}

template<typename T, int N> CUDA_FUNC_IN qMatrix<T, N, 1> solve(const qMatrix<T, N, N>& Q, const qMatrix<T, N, N>& R, const qMatrix<T, N, 1>& rhs)
{
	qMatrix<T, N, 1> b = Q.Transpose() * rhs;
	return solveUpperDiagonal(R, b);
}

//general purpose
template<typename T, int N> CUDA_FUNC_IN qMatrix<T, N, N> inv(const qMatrix<T, N, N>& A)
{
	qMatrix<T, N, N> L, U, P, I;
	luDecomposition(A, P, L, U);
	for(int i = 0; i < N; i++)
		I.col(i, solve(P, L, U, qMatrix<T, N, N>::e(i)));
	return I;
}

template<typename T, int N> CUDA_FUNC_IN T det(const qMatrix<T, N, N>& A)
{
	qMatrix<T, N, N> L, U, P;
	luDecomposition(A, P, L, U);
	T det = 1;
	for(int i = 0; i < N; i++)
		det *= L(i, i) * U(i, i);
	return det;
}

template<typename T, int N> CUDA_FUNC_IN qMatrix<T, N, 1> solve(const qMatrix<T, N, N>& A, const qMatrix<T, N, 1>& rhs)
{
	qMatrix<T, N, N> L, U, P;
	luDecomposition(A, P, L, U);
	return solve(P, L, U, rhs);
}

template<typename T, int M, int N> CUDA_FUNC_IN qMatrix<T, M, N> null(const qMatrix<T, N, M>& A, int& rank, const T& eps = T(1e-5))
{
	qMatrix<T, M, N> R;
	qMatrix<T, M, M> Q;
	qrHousholder(A, Q, R);
	rank = 0;
	while(rank < DMIN2(M, N) && std::abs(R(rank, rank)) > eps)
		rank++;
	qMatrix<T, M, N> nul = qMatrix<T, M, N>::Zero();
	for(int i = 0; i < rank; i++)
		nul.col(i, Q.col(N - 1 - rank + i));
	return nul;
}

template<typename T, int M, int N> CUDA_FUNC_IN int rank(const qMatrix<T, M, N>& A)
{
	int r;
	null(A, r);
	return r;
}

template<typename T, int N> CUDA_FUNC_IN void eig(const qMatrix<T, N, N>& A, qMatrix<T, N, N>& values, qMatrix<T, N, N>& vectors)
{
	qrAlgorithm(A, values, vectors);
	values = diagmat(values.diag());
}

template<typename T, int N> CUDA_FUNC_IN qMatrix<T, N, 1> eig(const qMatrix<T, N, N>& A)
{
	qMatrix<T, N, N> values, vectors;
	eig(A, values, vectors);
	return values.diag();
}

template<typename T, int M, int N> CUDA_FUNC_IN T cond(const qMatrix<T, M, N>& A)
{
	qMatrix<T, M, M> U;
	qMatrix<T, N, N> V;
	qMatrix<T, M, N> S;
	int n = svd(A, U, V, S);
	T la = 0, sm = 0;
	for(int i = 0; i < n; i++)
	{
		la = std::max(la, S(i, i));
		sm = std::min(sm, S(i, i));
	}
	return la / sm;
}

template<typename T, int N> CUDA_FUNC_IN T cond(const qMatrix<T, N, N>& A)
{
	return A.p_norm(T(2)) * inv(A).p_norm(T(2));
}