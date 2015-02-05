#pragma once

#include <MathTypes.h>

struct e_KernelFilterBase : public e_BaseType
{
	float xWidth, yWidth;
    float invXWidth, invYWidth;

	e_KernelFilterBase(float xw, float yw)
		: xWidth(xw), yWidth(yw), invXWidth(1.f/xw), invYWidth(1.f/yw)
	{

	}

	virtual void Update()
	{
		invXWidth = 1.0f / xWidth;
		invYWidth = 1.0f / yWidth;
	}

	CUDA_FUNC_IN e_KernelFilterBase()
		: xWidth(0), yWidth(0), invXWidth(0), invYWidth(0)
	{
	}
};

#define e_KernelBoxFilter_TYPE 1
struct e_KernelBoxFilter : public e_KernelFilterBase
{
	e_KernelBoxFilter(){}
	e_KernelBoxFilter(float xw, float yw)
		: e_KernelFilterBase(xw, yw)
	{

	}

	CUDA_FUNC_IN float Evaluate(float x, float y) const
	{
		return 1;
	}

	TYPE_FUNC(e_KernelBoxFilter)
};

#define e_KernelGaussianFilter_TYPE 2
struct e_KernelGaussianFilter : public e_KernelFilterBase
{
	float alpha;
    float expX, expY;
	
	e_KernelGaussianFilter(){}
	e_KernelGaussianFilter(float xw, float yw, float a)
		: e_KernelFilterBase(xw, yw), alpha(a), expX(expf(-alpha * xWidth * xWidth)),  expY(expf(-alpha * yWidth * yWidth))
	{

	}

	virtual void Update()
	{
		expX = expf(-alpha * xWidth * xWidth);
		expY = expf(-alpha * yWidth * yWidth);
	}

	CUDA_FUNC_IN float Gaussian(float d, float expv) const {
        return max(0.f, float(expf(-alpha * d * d) - expv));
    }

	CUDA_FUNC_IN float Evaluate(float x, float y) const
	{
		return Gaussian(x, expX) * Gaussian(y, expY);
	}

	TYPE_FUNC(e_KernelGaussianFilter)
};

#define e_KernelMitchellFilter_TYPE 3
struct e_KernelMitchellFilter : public e_KernelFilterBase
{
	float B, C;
	
	e_KernelMitchellFilter(){}
	e_KernelMitchellFilter(float b, float c, float xw, float yw)
		: e_KernelFilterBase(xw, yw), B(b), C(c)
	{

	}

	CUDA_FUNC_IN float Mitchell1D(float x) const {
        x = fabsf(2.f * x);
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

	TYPE_FUNC(e_KernelMitchellFilter)
};

#define e_KernelLanczosSincFilter_TYPE 4
struct e_KernelLanczosSincFilter : public e_KernelFilterBase
{
	float tau;
	
	e_KernelLanczosSincFilter(){}
	e_KernelLanczosSincFilter(float xw, float yw, float t)
		: e_KernelFilterBase(xw, yw), tau(t)
	{

	}

	CUDA_FUNC_IN float Sinc1D(float x) const {
        x = fabsf(x);
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

	TYPE_FUNC(e_KernelLanczosSincFilter)
};

#define e_KernelTriangleFilter_TYPE 5
struct e_KernelTriangleFilter : public e_KernelFilterBase
{
	e_KernelTriangleFilter(){}
	e_KernelTriangleFilter(float xw, float yw)
		: e_KernelFilterBase(xw, yw)
	{

	}

	CUDA_FUNC_IN float Evaluate(float x, float y) const
	{
		return max(0.f, xWidth - fabsf(x)) * max(0.f, yWidth - fabsf(y));
	}

	TYPE_FUNC(e_KernelTriangleFilter)
};

#define FLT_SIZE RND_16(Dmax5(sizeof(e_KernelBoxFilter), sizeof(e_KernelGaussianFilter), sizeof(e_KernelMitchellFilter), sizeof(e_KernelLanczosSincFilter), sizeof(e_KernelTriangleFilter)))

struct CUDA_ALIGN(16) e_KernelFilter : public e_AggregateBaseType<e_KernelFilterBase, FLT_SIZE>
{
public:
	CUDA_FUNC_IN e_KernelFilter()
	{
		//type = 0;	
	}

	CUDA_FUNC_IN float Evaluate(float x, float y) const
	{
		CALL_FUNC5(e_KernelBoxFilter,e_KernelGaussianFilter,e_KernelMitchellFilter,e_KernelLanczosSincFilter,e_KernelTriangleFilter, Evaluate(x, y))
		return 0.0f;
	}
};