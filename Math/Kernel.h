#pragma once

#include "MathFunc.h"
#include <Base/ValuePack.h>

namespace CudaTracerLib {

//the coefficient of the kernel volume
template<int DIM> CUDA_FUNC_IN static float c_d()
{
	return extract_val<DIM - 1>({ 2.0f, PI, 4.0f / 3.0f * PI });
}

template<int... DIMS> struct KernelBase
{
	//notation according to Silverman 1986 page 85
	//alpha = int_{-\infinity}^{\infinity}{t^2 K(t) \d{t}}
	//beta = int_{-\infinity}^{\infinity}{K(t)^2 \d{t}}
	//norm_factor = int_{-\infinity}^{\infinity}{K(t) \d{t}}

	//Mathematica code to compute the relevant constants
	/*
	Integrate1[f_] := Integrate[f@Abs[t], {t, -1, 1}];

	Integrate2[f_] := Integrate[f@Norm[{x, y}], {x, y} \[Element] Disk[]];

	Integrate3[f_] := Integrate[f@Norm[{x, y, z}], {x, y, z} \[Element] Ball[]];

	Eval[f_] := {Integrate1[f], Integrate2[f], Integrate3[f]};

	CNormFactor[K_] := Eval[K];

	CAlpha[K_] := Eval[#^2*K[#] &];

	CBeta[K_] := Eval[K[#]^2 &];
	*/
};

//Uniform := Function[{t}, Evaluate[1]];
struct UniformKernel : public KernelBase<1, 2, 3>
{
	template<int DIM> CUDA_FUNC_IN static float alpha()
	{
		return extract_val<DIM - 1>({ 2.0f / 3.0f, 1.0f / 2.0f * PI, 4.0f / 5.0f * PI });
	}

	template<int DIM> CUDA_FUNC_IN static float beta()
	{
		return c_d<DIM>();
	}

	template<int DIM> CUDA_FUNC_IN static float norm_factor()
	{
		return c_d<DIM>();
	}

	CUDA_FUNC_IN static float k(float t)
	{
		return 1.0f;
	}
};

//Perlin := Function[{t}, Evaluate[1 + t^3 (-6 t^2 + 15 t - 10)]];
struct PerlinKernel : public KernelBase<1, 2, 3>
{
	template<int DIM> CUDA_FUNC_IN static float alpha()
	{
		return extract_val<DIM - 1>({ 5.0f / 42.0f, 5.0f / 84.0f * PI, 1.0f / 15.0f * PI });
	}

	template<int DIM> CUDA_FUNC_IN static float beta()
	{
		return extract_val<DIM - 1>({ 181.0f / 231.0f, 41.0f / 231.0f * PI, 1070.0f / 9009.0f * PI });
	}

	template<int DIM> CUDA_FUNC_IN static float norm_factor()
	{
		return extract_val<DIM - 1>({ 1.0f, 2.0f / 7.0f * PI, 5.0f / 21.0f * PI });
	}

	CUDA_FUNC_IN static float k(float t)
	{
		return (1.0f + t * t * t * (-6.0f * t * t + 15.0f * t - 10.0f));
	}
};

//Triangular := Function[{t}, Evaluate[1-Abs[t]]];
struct TriangularKernel : public KernelBase<1, 2, 3>
{
	template<int DIM> CUDA_FUNC_IN static float alpha()
	{
		return extract_val<DIM - 1>({ 1.0f / 6.0f, 1.0f / 10.0f * PI, 2.0f / 15.0f * PI });
	}

	template<int DIM> CUDA_FUNC_IN static float beta()
	{
		return extract_val<DIM - 1>({ 2.0f / 3.0f, 1.0f / 6.0f * PI, 2.0f / 15.0f * PI });
	}

	template<int DIM> CUDA_FUNC_IN static float norm_factor()
	{
		return extract_val<DIM - 1>({ 1.0f, 1.0f / 3.0f * PI, 1.0f / 3.0f * PI });
	}

	CUDA_FUNC_IN static float k(float t)
	{
		return 1.0f - math::abs(t);
	}
};

//Epanechnikov := Function[{t}, Evaluate[3.0/4.0*(1-t^2)]];
struct EpanechnikovKernel : public KernelBase<1, 2, 3>
{
	template<int DIM> CUDA_FUNC_IN static float alpha()
	{
		return extract_val<DIM - 1>({ 0.2f, 0.392699f, 0.538559f });
	}

	template<int DIM> CUDA_FUNC_IN static float beta()
	{
		return extract_val<DIM - 1>({ 0.6f, 0.589049f, 0.538559f });
	}

	template<int DIM> CUDA_FUNC_IN static float norm_factor()
	{
		return extract_val<DIM - 1>({ 1.0f, 1.1781f, 1.25664f });
	}

	CUDA_FUNC_IN static float k(float t)
	{
		return 3.0f / 4.0f * (1 - t * t);
	}
};

//Quartic := Function[{t}, Evaluate[15.0/16.0*(1-t^2)^2]];
struct QuarticKernel : public KernelBase<1, 2, 3>
{
	template<int DIM> CUDA_FUNC_IN static float alpha()
	{
		return extract_val<DIM - 1>({ 0.142857f, 0.245437f, 0.299199f });
	}

	template<int DIM> CUDA_FUNC_IN static float beta()
	{
		return extract_val<DIM - 1>({ 0.714286f, 0.552233f, 0.407999f });
	}

	template<int DIM> CUDA_FUNC_IN static float norm_factor()
	{
		return extract_val<DIM - 1>({ 1.0f, 0.981748f, 0.897598f });
	}

	CUDA_FUNC_IN static float k(float t)
	{
		return 15.0f / 16.0f * math::sqr(1 - t * t);
	}
};

//Triweight := Function[{t}, Evaluate[35.0/32.0*(1-t^2)^3]];
struct TriweightKernel : public KernelBase<1, 2, 3>
{
	template<int DIM> CUDA_FUNC_IN static float alpha()
	{
		return extract_val<DIM - 1>({ 0.111111f, 0.171806f, 0.1904f });
	}

	template<int DIM> CUDA_FUNC_IN static float beta()
	{
		return extract_val<DIM - 1>({ 0.815851f, 0.536893f, 0.341743f });
	}

	template<int DIM> CUDA_FUNC_IN static float norm_factor()
	{
		return extract_val<DIM - 1>({ 1.0f, 0.859029f, 0.698132f });
	}

	CUDA_FUNC_IN static float k(float t)
	{
		return 35.0f / 32.0f * pow_int_compile<3>::pow(1 - t * t);
	}
};

//Tricube := Function[{t}, Evaluate[70.0/81.0 * (1 - Abs[t]^3)^3]];
struct TricubeKernel : public KernelBase<1, 2, 3>
{
	template<int DIM> CUDA_FUNC_IN static float alpha()
	{
		return extract_val<DIM - 1>({ 0.144033f, 0.241661f, 0.285599f });
	}

	template<int DIM> CUDA_FUNC_IN static float beta()
	{
		return extract_val<DIM - 1>({ 0.708502f, 0.587999f, 0.446906f });
	}

	template<int DIM> CUDA_FUNC_IN static float norm_factor()
	{
		return extract_val<DIM - 1>({ 1.0f, 0.999598f, 0.904986f });
	}

	CUDA_FUNC_IN static float k(float t)
	{
		return 70.0f / 81.0f * pow_int_compile<3>::pow(1 - pow_int_compile<3>::pow(math::abs(t)));
	}
};

//Gaussian := Function[{t}, Evaluate[1.0 / (Sqrt[2.0*PI]) * Exp[-1.0 / 2.0*t ^ 2]]];
struct GaussianKernel : public KernelBase<1, 2, 3>
{
	template<int DIM> CUDA_FUNC_IN static float alpha()
	{
		return extract_val<DIM - 1>({ 0.35227173435040204 / math::sqrt(PI), 0.8015317177951413 / math::sqrt(PI), 1.2506763172009567 / math::sqrt(PI) });
	}

	template<int DIM> CUDA_FUNC_IN static float beta()
	{
		return extract_val<DIM - 1>({ 0.7468241328124271 / PI, 0.9929326518994357 / PI, 1.190489859376167 / PI });
	}

	template<int DIM> CUDA_FUNC_IN static float norm_factor()
	{
		return extract_val<DIM - 1>({ 1.2100356193111086f / math::sqrt(PI), 1.7481382186511691f / math::sqrt(PI), 2.213388585405117f / math::sqrt(PI) });
	}

	CUDA_FUNC_IN static float k(float t)
	{
		return 1.0f / math::sqrt(2.0f * PI) * math::exp(-1.0f / 2.0f * t * t);
	}
};

//Cosine := Function[{t}, Evaluate[Pi/4.0*Cos[Pi/2.0*t]]];
struct CosineKernel : public KernelBase<1, 2, 3>
{
	template<int DIM> CUDA_FUNC_IN static float alpha()
	{
		return extract_val<DIM - 1>({ 0.189431f, 0.365572f, 0.494615f });
	}

	template<int DIM> CUDA_FUNC_IN static float beta()
	{
		return extract_val<DIM - 1>({ 0.61685f, 0.576247f, 0.50653f });
	}

	template<int DIM> CUDA_FUNC_IN static float norm_factor()
	{
		return extract_val<DIM - 1>({ 1.0f, 1.14159f, 1.19023f });
	}

	CUDA_FUNC_IN static float k(float t)
	{
		return PI / 4.0f * math::cos(PI / 2.0f * t);
	}
};

//Silverman := Function[{t}, Evaluate[1.0 / 2.0*Exp[-Abs[t] / Sqrt[2]] * Sin[Abs[t] / Sqrt[2] + Pi / 4]]];
struct SilvermanKernel : public KernelBase<1, 2, 3>
{
	template<int DIM> CUDA_FUNC_IN static float alpha()
	{
		return extract_val<DIM - 1>({ 0.188651f, 0.433726f, 0.682024f });
	}

	template<int DIM> CUDA_FUNC_IN static float beta()
	{
		return extract_val<DIM - 1>({ 0.197611f, 0.273905f, 0.338403f });
	}

	template<int DIM> CUDA_FUNC_IN static float norm_factor()
	{
		return extract_val<DIM - 1>({ 0.625147f, 0.922657f, 1.18533f });
	}

	CUDA_FUNC_IN static float k(float t)
	{
		return 1.0f / 2.0f * math::exp(-math::abs(t) / math::sqrt(2)) * math::sin(math::abs(t) / math::sqrt(2.0f) + PI / 4.0f);
	}
};

template<typename K> struct KernelWrapper
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

	template<int DIM> CUDA_FUNC_IN static float alpha()
	{
		return K::template alpha<DIM>() / K::template norm_factor<DIM>();
	}

	template<int DIM> CUDA_FUNC_IN static float beta()
	{
		return K::template beta<DIM>() / math::sqr(K::template norm_factor<DIM>());
	}
};

}