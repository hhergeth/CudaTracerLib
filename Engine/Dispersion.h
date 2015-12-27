#pragma once

#include <MathTypes.h>
#include <VirtualFuncType.h>
#include "Samples.h"

namespace CudaTracerLib
{

struct DispersionBase : public BaseType
{

};

struct DispersionCauchy : public DispersionBase
{
	TYPE_FUNC(1)
	float B, C;

	DispersionCauchy(float b = 1.49f, float c = 0.0f)
		: B(b), C(c)
	{

	}

	CUDA_FUNC_IN float calc_eta(float lambda) const
	{
		return B + C / (lambda / 1e3);
	}

	CUDA_FUNC_IN float calc_lambda(float eta) const
	{
		return 1000 * C / (eta - B);
	}

	CUDA_FUNC_IN bool hasDispersion() const
	{
		return C != 0.0f;
	}
};

struct DispersionSellmeier : public DispersionBase
{
	TYPE_FUNC(2)
	float A;
	Vec3f B, C;

	DispersionSellmeier(float a = 1, Vec3f b = Vec3f(0), Vec3f c = Vec3f(0))
		: A(a), B(b), C(c)
	{

	}

	CUDA_FUNC_IN float calc_eta(float lambda) const
	{
		float l = lambda / 1000.0f, l2 = l * l;
		float eta2 = A + (B.x * l2) / (l2 - C.x) + (B.y * l2) / (l2 - C.y) + (B.z * l2) / (l2 - C.z);
		return math::sqrt(eta2);
	}

	CUDA_FUNC_IN float calc_lambda(float eta) const
	{
		//bad approximation, wolfram alpha's analytic solution seems rather complicated
		float b = B.sum() / 3.0f, c = C.sum() / 3.0f;
		float p = (eta * eta - A) / b, q = c;
		float l2 = p * q / (p - 1);
		return math::sqrt(l2);
	}

	CUDA_FUNC_IN bool hasDispersion() const
	{
		return B != Vec3f(0) || C != Vec3f(0);
	}
};

struct DispersionLinear : public DispersionBase
{
	TYPE_FUNC(3)
	float minEta, maxEta;

	DispersionLinear(float min = 1.49f, float max = 1.49f)
		: minEta(min), maxEta(max)
	{

	}

	CUDA_FUNC_IN float calc_eta(float lambda) const
	{
		return math::lerp(minEta, maxEta, (lambda - 300) / (900.0f - 300.0f));
	}

	CUDA_FUNC_IN float calc_lambda(float eta) const
	{
		return math::lerp(300, 900, (eta - minEta) / (maxEta - minEta));
	}

	CUDA_FUNC_IN bool hasDispersion() const
	{
		return minEta != maxEta;
	}
};

struct Dispersion : public CudaVirtualAggregate<DispersionBase, DispersionLinear, DispersionCauchy, DispersionSellmeier>
{
	CALLER(hasDispersion)
	CUDA_FUNC_IN bool hasDispersion() const
	{
		return hasDispersion_Caller<bool>(*this);
	}

	CALLER(calc_eta)
	CUDA_FUNC_IN float calc_eta(float lambda) const
	{
		return calc_eta_Caller<float>(*this, lambda);
	}

	CALLER(calc_lambda)
	CUDA_FUNC_IN float calc_lambda(float eta) const
	{
		return calc_lambda_Caller<float>(*this, eta);
	}

	//always returns eta and f/pdf or f or pdf in parameter

	CUDA_FUNC_IN float sample_eta(const BSDFSamplingRecord& bRec, float sample, Spectrum& f_o, float& pdf) const
	{
		if (!hasDispersion())
		{
			f_o = Spectrum(1.0f);
			pdf = 1.0f;
			return calc_eta(600);
		}

		float w = bRec.f_i.SampleWavelength(f_o, pdf, sample);//wavelength in nm
		for (int i = 0; i < SPECTRUM_SAMPLES; i++)
			f_o[i] = bRec.f_i[i] != 0 ? f_o[i] / bRec.f_i[i] : 0;
		return calc_eta(w);
	}

	CUDA_FUNC_IN float f_eta(const BSDFSamplingRecord& bRec, Spectrum& f_o) const
	{
		if (!hasDispersion())
		{
			f_o = Spectrum(1.0f);
			return calc_eta(600);
		}

		float eta = Frame::sinTheta(bRec.wi) / Frame::sinTheta(bRec.wo);
		float lambda = calc_lambda(eta);
		f_o = bRec.f_i.FWavelength(lambda);
		for (int i = 0; i < SPECTRUM_SAMPLES; i++)
			f_o[i] = bRec.f_i[i] != 0 ? f_o[i] / bRec.f_i[i] : 0;
		return eta;
	}

	CUDA_FUNC_IN float pdf_eta(const BSDFSamplingRecord& bRec, float& pdf) const
	{
		if (!hasDispersion())
		{
			pdf = 1.0f;
			return calc_eta(600);
		}

		float eta = Frame::sinTheta(bRec.wi) / Frame::sinTheta(bRec.wo);
		float lambda = calc_lambda(eta);
		pdf = bRec.f_i.PdfWavelength(lambda);
		return eta;
	}
};

}