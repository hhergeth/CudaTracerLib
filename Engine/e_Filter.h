#pragma once

#include <MathTypes.h>

struct e_KernelFilterBase
{
	const float xWidth, yWidth;
    const float invXWidth, invYWidth;

	e_KernelFilterBase(float xw, float yw)
		: xWidth(xw), yWidth(yw), invXWidth(1.f/xw), invYWidth(1.f/yw)
	{

	}

	CUDA_FUNC_IN e_KernelFilterBase()
		: xWidth(0), yWidth(0), invXWidth(0), invYWidth(0)
	{
	}
};

#define e_KernelBoxFilter_TYPE 1
struct e_KernelBoxFilter : public e_KernelFilterBase
{
	e_KernelBoxFilter(float xw, float yw)
		: e_KernelFilterBase(xw, yw)
	{

	}

	e_KernelBoxFilter& operator=(const e_KernelBoxFilter& element)
	{
		return e_KernelBoxFilter(element.xWidth, element.yWidth);
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
	const float alpha;
    const float expX, expY;

	e_KernelGaussianFilter(float xw, float yw, float a)
		: e_KernelFilterBase(xw, yw), alpha(a), expX(expf(-alpha * xWidth * xWidth)),  expY(expf(-alpha * yWidth * yWidth))
	{

	}

	e_KernelGaussianFilter& operator=(const e_KernelGaussianFilter& element)
	{
		return e_KernelGaussianFilter(element.xWidth, element.yWidth, element.alpha);
	}

	CUDA_FUNC_IN float Gaussian(float d, float expv) const {
        return MAX(0.f, float(expf(-alpha * d * d) - expv));
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
	const float B, C;

	e_KernelMitchellFilter(float b, float c, float xw, float yw)
		: e_KernelFilterBase(xw, yw), B(b), C(c)
	{

	}

	e_KernelMitchellFilter& operator=(const e_KernelMitchellFilter& element)
	{
		return e_KernelMitchellFilter(element.B, element.C, element.xWidth, element.yWidth);
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
	const float tau;

	e_KernelLanczosSincFilter(float xw, float yw, float t)
		: e_KernelFilterBase(xw, yw), tau(t)
	{

	}

	e_KernelLanczosSincFilter& operator=(const e_KernelLanczosSincFilter& element)
	{
		return e_KernelLanczosSincFilter(element.xWidth, element.yWidth, element.tau);
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
	e_KernelTriangleFilter(float xw, float yw)
		: e_KernelFilterBase(xw, yw)
	{

	}

	e_KernelTriangleFilter& operator=(const e_KernelTriangleFilter& element)
	{
		return e_KernelTriangleFilter(element.xWidth, element.yWidth);
	}

	CUDA_FUNC_IN float Evaluate(float x, float y) const
	{
		return MAX(0.f, xWidth - fabsf(x)) * MAX(0.f, yWidth - fabsf(y));
	}

	TYPE_FUNC(e_KernelTriangleFilter)
};

#define FLT_SIZE RND_16(DMAX5(sizeof(e_KernelBoxFilter), sizeof(e_KernelGaussianFilter), sizeof(e_KernelMitchellFilter), sizeof(e_KernelLanczosSincFilter), sizeof(e_KernelTriangleFilter)))

struct CUDA_ALIGN(16) e_KernelFilter
{
private:
	CUDA_ALIGN(16) unsigned char Data[FLT_SIZE];
	unsigned int type;
#define CALL_TYPE(t,f,r) \
	case t##_TYPE : \
		r ((t*)Data)->f; \
		break;
#define CALL_FUNC(r,f) \
	switch (type) \
	{ \
		CALL_TYPE(e_KernelBoxFilter, f, r) \
		CALL_TYPE(e_KernelGaussianFilter, f, r) \
		CALL_TYPE(e_KernelMitchellFilter, f, r) \
		CALL_TYPE(e_KernelLanczosSincFilter, f, r) \
		CALL_TYPE(e_KernelTriangleFilter, f, r) \
	}
public:
	CUDA_FUNC_IN e_KernelFilter()
	{
		//type = 0;	
	}

	CUDA_FUNC_IN float Evaluate(float x, float y) const
	{
		CALL_FUNC(return, Evaluate(x, y))
	}

	template<typename T> CUDA_FUNC_IN T* As()
	{
		return (T*)Data;
	}

	template<typename T> void SetData(const T& val)
	{
		memcpy(Data, &val, sizeof(T));
		type = T::TYPE();
	}
#undef CALL_FUNC
#undef CALL_TYPE
};