#pragma once

#include "MathFunc.h"
#include <Base/ValuePack.h>

namespace CudaTracerLib {
	
	template<int DIM> struct pow_int_compile
	{
		CUDA_FUNC_IN static float pow(float f)
		{
			return f * pow_int_compile<DIM - 1>::pow(f);
		}
	};

	template<> struct pow_int_compile<0>
	{
		CUDA_FUNC_IN static float pow(float f)
		{
			return 1.0f;
		}
	};

	template<int... DIMS> struct KernelBase
	{
		//the coefficient of the kernel volume
		template<int DIM> CUDA_FUNC_IN static float c_d()
		{
			return extract_val<DIM - 1>(2.0f, PI, 4.0f / 3.0f * PI);
		}

		//notation according to Silverman 1986 page 85
		//alpha = int_{-\infinity}^{\infinity}{t^2 K(t) \d{t}}
		//beta = int_{-\infinity}^{\infinity}{K(t)^2 \d{t}}
		//norm_factor = int_{-\infinity}^{\infinity}{K(t) \d{t}} / volume
	};

	struct UniformKernel : public KernelBase<1, 2, 3>
	{
		template<int DIM> CUDA_FUNC_IN static float alpha()
		{
			return extract_val<DIM - 1>(1.0f / 3.0f, 0.785398f, 1.25664f);
		}

		template<int DIM> CUDA_FUNC_IN static float beta()
		{
			return extract_val<DIM - 1>(0.5f, 0.785398f, 1.0472);
		}

		template<int DIM> CUDA_FUNC_IN static float norm_factor()
		{
			return extract_val<DIM - 1>(1.0f, 1.0f, 1.0f);
		}

		CUDA_FUNC_IN static float k(float t)
		{
			return 1.0f;
		}
	};

	struct PerlinKernel : public KernelBase<1, 2, 3>
	{
		template<int DIM> CUDA_FUNC_IN static float alpha()
		{
			return extract_val<DIM - 1>(5.0f / 42.0f, 5.0f / 84.0f * PI, 1.0f / 15.0f * PI);
		}

		template<int DIM> CUDA_FUNC_IN static float beta()
		{
			return extract_val<DIM - 1>(181.0f / 231.0f, 41.0f / 231.0f * PI, 1070.0f / 9009.0f * PI);
		}

		template<int DIM> CUDA_FUNC_IN static float norm_factor()
		{
			return extract_val<DIM - 1>(1.0f, 2.0f / 7.0f * PI, 2.0f / 3.0f * PI);
		}

		CUDA_FUNC_IN static float k(float t)
		{
			return (1.0f + t * t * t * (-6.0f * t * t + 15.0f * t - 10.0f));
		}
	};

	template<typename K> struct KernelWrapper : public K
	{
		template<int DIM> CUDA_FUNC_IN static float k(float t, float r)
		{
			const float vol = pow_int_compile<DIM>::pow(r);
			const float norm_coeff = K::template norm_factor<DIM>();
			return K::k(math::clamp01(t / r)) / (norm_coeff * vol);
		}

		template<int DIM, typename VEC> CUDA_FUNC_IN static float k(const VEC& t, float r)
		{
			return k<DIM>(length(t), r);
		}
	};

}