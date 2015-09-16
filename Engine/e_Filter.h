#pragma once

#include <MathTypes.h>
#include "../VirtualFuncType.h"

//Implementation and interface designed after PBRT.

struct e_FilterBase : public e_BaseType//, public e_BaseTypeHelper<5524550>
{
	float xWidth, yWidth;
    float invXWidth, invYWidth;

	e_FilterBase(float xw, float yw)
		: xWidth(xw), yWidth(yw), invXWidth(1.f/xw), invYWidth(1.f/yw)
	{

	}

	virtual void Update()
	{
		invXWidth = 1.0f / xWidth;
		invYWidth = 1.0f / yWidth;
	}

	CUDA_FUNC_IN e_FilterBase()
		: xWidth(0), yWidth(0), invXWidth(0), invYWidth(0)
	{
	}
};

struct e_BoxFilter : public e_FilterBase//, public e_DerivedTypeHelper<1>
{
	TYPE_FUNC(1)
	e_BoxFilter(){}
	e_BoxFilter(float xw, float yw)
		: e_FilterBase(xw, yw)
	{

	}

	CUDA_FUNC_IN float Evaluate(float x, float y) const
	{
		return 1;
	}
};

struct e_GaussianFilter : public e_FilterBase//, public e_DerivedTypeHelper<2>
{
	TYPE_FUNC(2)
	float alpha;
    float expX, expY;
	
	e_GaussianFilter(){}
	e_GaussianFilter(float xw, float yw, float a)
		: e_FilterBase(xw, yw), alpha(a), expX(expf(-alpha * xWidth * xWidth)),  expY(expf(-alpha * yWidth * yWidth))
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

struct e_MitchellFilter : public e_FilterBase//, public e_DerivedTypeHelper<3>
{
	TYPE_FUNC(3)
	float B, C;
	
	e_MitchellFilter(){}
	e_MitchellFilter(float b, float c, float xw, float yw)
		: e_FilterBase(xw, yw), B(b), C(c)
	{

	}

	CUDA_FUNC_IN float Mitchell1D(float x) const {
        x = math::abs(2.f * x);
        if (x > 1.f)
            return ((-B - 6*C) * x*x*x + (6*B + 30*C) * x*x +
                    (-12*B - 48*C) * x + (8*B + 24*C)) * (1.f/6.f);
        else
            return ((12 - 9*B - 6*C) * x*x*x +
                    (-18 + 12*B + 6*C) * x*x +
                    (6 - 2*B)) * (1.f/6.f);
    }

	CUDA_FUNC_IN float Evaluate(float x, float y) const
	{
		return Mitchell1D(x * invXWidth) * Mitchell1D(y * invYWidth);
	}
};

struct e_LanczosSincFilter : public e_FilterBase//, public e_DerivedTypeHelper<4>
{
	TYPE_FUNC(4)
	float tau;
	
	e_LanczosSincFilter(){}
	e_LanczosSincFilter(float xw, float yw, float t)
		: e_FilterBase(xw, yw), tau(t)
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

struct e_TriangleFilter : public e_FilterBase//, public e_DerivedTypeHelper<5>
{
	TYPE_FUNC(5)
	e_TriangleFilter(){}
	e_TriangleFilter(float xw, float yw)
		: e_FilterBase(xw, yw)
	{

	}

	CUDA_FUNC_IN float Evaluate(float x, float y) const
	{
		return max(0.f, xWidth - math::abs(x)) * max(0.f, yWidth - math::abs(y));
	}
};

struct CUDA_ALIGN(16) e_Filter : public CudaVirtualAggregate<e_FilterBase, e_BoxFilter, e_GaussianFilter, e_MitchellFilter, e_LanczosSincFilter, e_TriangleFilter>
{
public:
	CUDA_FUNC_IN e_Filter()
	{
		//type = 0;	
	}

	CALLER(Evaluate)
	CUDA_FUNC_IN float Evaluate(float x, float y) const
	{
		return Evaluate_Caller<float>(*this, x, y);
	}
};