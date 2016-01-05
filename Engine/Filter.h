#pragma once

#include <MathTypes.h>
#include <VirtualFuncType.h>

//Implementation and interface designed after PBRT.

namespace CudaTracerLib {

struct FilterBase : public BaseType//, public BaseTypeHelper<5524550>
{
	float xWidth, yWidth;
	float invXWidth, invYWidth;

	CUDA_FUNC_IN FilterBase(float xw, float yw)
		: xWidth(xw), yWidth(yw), invXWidth(1.f / xw), invYWidth(1.f / yw)
	{

	}

	virtual void Update()
	{
		invXWidth = 1.0f / xWidth;
		invYWidth = 1.0f / yWidth;
	}

	CUDA_FUNC_IN FilterBase()
		: xWidth(0), yWidth(0), invXWidth(0), invYWidth(0)
	{
	}
};

struct BoxFilter : public FilterBase//, public e_DerivedTypeHelper<1>
{
	TYPE_FUNC(1)
	BoxFilter(){}
	BoxFilter(float xw, float yw)
		: FilterBase(xw, yw)
	{

	}

	CUDA_FUNC_IN float Evaluate(float x, float y) const
	{
		return 1;
	}
};

struct GaussianFilter : public FilterBase//, public e_DerivedTypeHelper<2>
{
	TYPE_FUNC(2)
	float alpha;
	float expX, expY;

	GaussianFilter(){}
	CUDA_FUNC_IN GaussianFilter(float xw, float yw, float a)
		: FilterBase(xw, yw), alpha(a), expX(expf(-alpha * xWidth * xWidth)), expY(expf(-alpha * yWidth * yWidth))
	{

	}

	virtual void Update()
	{
		expX = math::exp(-alpha * xWidth * xWidth);
		expY = math::exp(-alpha * yWidth * yWidth);
	}

	CUDA_FUNC_IN float Gaussian(float d, float expv) const {
		return max(0.f, float(math::exp(-alpha * d * d) - expv));
	}

	CUDA_FUNC_IN float Evaluate(float x, float y) const
	{
		return Gaussian(x, expX) * Gaussian(y, expY);
	}
};

struct MitchellFilter : public FilterBase//, public e_DerivedTypeHelper<3>
{
	TYPE_FUNC(3)
	float B, C;

	MitchellFilter(){}
	MitchellFilter(float b, float c, float xw, float yw)
		: FilterBase(xw, yw), B(b), C(c)
	{

	}

	CUDA_FUNC_IN float Mitchell1D(float x) const {
		x = math::abs(2.f * x);
		if (x > 1.f)
			return ((-B - 6 * C) * x*x*x + (6 * B + 30 * C) * x*x +
			(-12 * B - 48 * C) * x + (8 * B + 24 * C)) * (1.f / 6.f);
		else
			return ((12 - 9 * B - 6 * C) * x*x*x +
			(-18 + 12 * B + 6 * C) * x*x +
			(6 - 2 * B)) * (1.f / 6.f);
	}

	CUDA_FUNC_IN float Evaluate(float x, float y) const
	{
		return Mitchell1D(x * invXWidth) * Mitchell1D(y * invYWidth);
	}
};

struct LanczosSincFilter : public FilterBase//, public e_DerivedTypeHelper<4>
{
	TYPE_FUNC(4)
	float tau;

	LanczosSincFilter(){}
	LanczosSincFilter(float xw, float yw, float t)
		: FilterBase(xw, yw), tau(t)
	{

	}

	CUDA_FUNC_IN float Sinc1D(float x) const {
		x = math::abs(x);
		if (x < 1e-5) return 1.f;
		if (x > 1.)   return 0.f;
		x *= PI;
		float sinc = sinf(x) / x;
		float lanczos = sinf(x * tau) / (x * tau);
		return sinc * lanczos;
	}

	CUDA_FUNC_IN float Evaluate(float x, float y) const
	{
		return Sinc1D(x * invXWidth) * Sinc1D(y * invYWidth);
	}
};

struct TriangleFilter : public FilterBase//, public e_DerivedTypeHelper<5>
{
	TYPE_FUNC(5)
	TriangleFilter(){}
	TriangleFilter(float xw, float yw)
		: FilterBase(xw, yw)
	{

	}

	CUDA_FUNC_IN float Evaluate(float x, float y) const
	{
		return max(0.f, xWidth - math::abs(x)) * max(0.f, yWidth - math::abs(y));
	}
};

struct CUDA_ALIGN(16) Filter : public CudaVirtualAggregate<FilterBase, BoxFilter, GaussianFilter, MitchellFilter, LanczosSincFilter, TriangleFilter>
{
public:
	CUDA_FUNC_IN Filter()
	{
		//type = 0;	
	}

	CALLER(Evaluate)
		CUDA_FUNC_IN float Evaluate(float x, float y) const
	{
		return Evaluate_Caller<float>(*this, x, y);
	}
};

}