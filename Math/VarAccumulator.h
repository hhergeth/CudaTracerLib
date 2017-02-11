#pragma once

#include "MathFunc.h"

namespace CudaTracerLib {

template<typename T> CUDA_FUNC_IN T VarianceFromMoments(const T& SUM_X, const T& SUM_X2, float N)
{
	float invN = 1.0f / N;
	return (SUM_X2 - math::sqr(SUM_X) * invN) * invN;
}

template<typename T> struct VarAccumulator
{
	T Sum_X;
	T Sum_X2;

	CUDA_FUNC_IN VarAccumulator()
		: Sum_X(0), Sum_X2(0)
	{

	}

	CUDA_FUNC_IN VarAccumulator(const T& sum, const T& sum2)
		: Sum_X(sum), Sum_X2(sum2)
	{

	}

	CUDA_FUNC_IN VarAccumulator& operator+=(const T& rhs)
	{
		Sum_X += rhs;
		Sum_X2 += math::sqr(rhs);
		return *this;
	}

	CUDA_FUNC_IN void Add(const T& X, const T& X2)
	{
		Sum_X += X;
		Sum_X2 += X2;
	}

	CUDA_FUNC_IN T E(float sampleSize) const
	{
		return Sum_X / sampleSize;
	}

	CUDA_FUNC_IN T Var(float sampleSize) const
	{
		return VarianceFromMoments(Sum_X, Sum_X2, sampleSize);
	}

	CUDA_FUNC_IN VarAccumulator operator+(const VarAccumulator& rhs) const
	{
		return VarAccumulator(Sum_X + rhs.Sum_X, Sum_X2 + rhs.Sum_X2);
	}

	CUDA_FUNC_IN VarAccumulator operator-(const VarAccumulator& rhs) const
	{
		return VarAccumulator(Sum_X - rhs.Sum_X, Sum_X2 - rhs.Sum_X2);
	}

	CUDA_FUNC_IN VarAccumulator operator*(const VarAccumulator& rhs) const
	{
		return VarAccumulator(Sum_X * rhs.Sum_X, Sum_X2 * rhs.Sum_X2);
	}

	CUDA_FUNC_IN VarAccumulator operator*(float rhs) const
	{
		return VarAccumulator(Sum_X * rhs, Sum_X2 * rhs * rhs);
	}

	CUDA_FUNC_IN VarAccumulator operator/(float rhs) const
	{
		return this->operator*(1.0f / rhs);
	}
};

}