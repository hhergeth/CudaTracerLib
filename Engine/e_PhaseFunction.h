#pragma once

#include "..\Math\AABB.h"
#include "..\Base\CudaRandom.h"
#include "e_Samples.h"

//this architecture and the implementations are completly copied from mitsuba!

enum EPhaseFunctionType
{
	/// Completely isotropic 1/(4 pi) phase function
	pEIsotropic       = 0x01,
	/// The phase function only depends on \c dot(wi,wo)
	pEAngleDependence = 0x04,
	/// The opposite of \ref EAngleDependence (there is an arbitrary dependence)
	pEAnisotropic     = 0x02,
	/// The phase function is non symmetric, i.e. eval(wi,wo) != eval(wo, wi)
	pENonSymmetric    = 0x08
};

struct e_BasePhaseFunction : public e_BaseType
{
	EPhaseFunctionType type;

	e_BasePhaseFunction(){}
	e_BasePhaseFunction(EPhaseFunctionType t)
		: type(t)
	{

	}
};

#define e_HGPhaseFunction_TYPE 1
struct e_HGPhaseFunction : public e_BasePhaseFunction
{
	float m_g;

	e_HGPhaseFunction(){}
	e_HGPhaseFunction(float g)
		: e_BasePhaseFunction(EPhaseFunctionType::pEAngleDependence), m_g(g)
	{
	}

	CUDA_DEVICE CUDA_HOST float Evaluate(const PhaseFunctionSamplingRecord &pRec) const;

	CUDA_DEVICE CUDA_HOST float Sample(PhaseFunctionSamplingRecord &pRec, CudaRNG& sampler) const;

	CUDA_DEVICE CUDA_HOST float Sample(PhaseFunctionSamplingRecord &pRec, float &pdf, CudaRNG& sampler) const;

	TYPE_FUNC(e_HGPhaseFunction)
};

#define e_IsotropicPhaseFunction_TYPE 2
struct e_IsotropicPhaseFunction : public e_BasePhaseFunction
{
	e_IsotropicPhaseFunction()
		: e_BasePhaseFunction((EPhaseFunctionType)(pEIsotropic | pEAngleDependence))
	{
	}

	CUDA_DEVICE CUDA_HOST float Evaluate(const PhaseFunctionSamplingRecord &pRec) const;

	CUDA_DEVICE CUDA_HOST float Sample(PhaseFunctionSamplingRecord &pRec, CudaRNG& sampler) const;

	CUDA_DEVICE CUDA_HOST float Sample(PhaseFunctionSamplingRecord &pRec, float &pdf, CudaRNG& sampler) const;

	TYPE_FUNC(e_IsotropicPhaseFunction)
};

#define e_KajiyaKayPhaseFunction_TYPE 3
struct e_KajiyaKayPhaseFunction : public e_BasePhaseFunction
{
	float m_ks, m_kd, m_exponent, m_normalization;
	float3 orientation;
	
	e_KajiyaKayPhaseFunction(){}
	e_KajiyaKayPhaseFunction(float ks, float kd, float e, float3 o);

	virtual void Update();

	CUDA_DEVICE CUDA_HOST float Evaluate(const PhaseFunctionSamplingRecord &pRec) const;

	CUDA_DEVICE CUDA_HOST float Sample(PhaseFunctionSamplingRecord &pRec, CudaRNG& sampler) const;

	CUDA_DEVICE CUDA_HOST float Sample(PhaseFunctionSamplingRecord &pRec, float &pdf, CudaRNG& sampler) const;

	TYPE_FUNC(e_KajiyaKayPhaseFunction)
};

#define e_RayleighPhaseFunction_TYPE 4
struct e_RayleighPhaseFunction : public e_BasePhaseFunction
{
	e_RayleighPhaseFunction()
		: e_BasePhaseFunction(EPhaseFunctionType::pEIsotropic)
	{
	}

	CUDA_DEVICE CUDA_HOST float Evaluate(const PhaseFunctionSamplingRecord &pRec) const;

	CUDA_DEVICE CUDA_HOST float Sample(PhaseFunctionSamplingRecord &pRec, CudaRNG& sampler) const;

	CUDA_DEVICE CUDA_HOST float Sample(PhaseFunctionSamplingRecord &pRec, float &pdf, CudaRNG& sampler) const;

	TYPE_FUNC(e_RayleighPhaseFunction)
};

#define PHF_SIZE RND_16(DMAX4(sizeof(e_HGPhaseFunction), sizeof(e_IsotropicPhaseFunction), sizeof(e_KajiyaKayPhaseFunction), sizeof(e_RayleighPhaseFunction)))

struct CUDA_ALIGN(16) e_PhaseFunction : public e_AggregateBaseType<e_BasePhaseFunction, PHF_SIZE>
{
public:
	e_PhaseFunction()
	{
		type = 0;
	}

	CUDA_FUNC_IN EPhaseFunctionType getType() const
	{
		return ((e_BasePhaseFunction*)Data)->type;
	}

	CUDA_FUNC_IN float Evaluate(const PhaseFunctionSamplingRecord &pRec) const
	{
		CALL_FUNC4(e_HGPhaseFunction,e_IsotropicPhaseFunction,e_KajiyaKayPhaseFunction,e_RayleighPhaseFunction, Evaluate(pRec))
		return 0.0f;
	}

	CUDA_FUNC_IN float Sample(PhaseFunctionSamplingRecord &pRec, CudaRNG& sampler) const
	{
		CALL_FUNC4(e_HGPhaseFunction,e_IsotropicPhaseFunction,e_KajiyaKayPhaseFunction,e_RayleighPhaseFunction, Sample(pRec, sampler))
		return 0.0f;
	}

	CUDA_FUNC_IN float Sample(PhaseFunctionSamplingRecord &pRec, float &pdf, CudaRNG& sampler) const
	{
		CALL_FUNC4(e_HGPhaseFunction,e_IsotropicPhaseFunction,e_KajiyaKayPhaseFunction,e_RayleighPhaseFunction, Sample(pRec, pdf, sampler))
		return 0.0f;
	}

	CUDA_FUNC_IN float pdf(const PhaseFunctionSamplingRecord &pRec) const
	{
		return Evaluate(pRec);
	}
};

template<typename T> e_PhaseFunction CreatePhaseFunction(const T& val)
{
	e_PhaseFunction r;
	r.SetData(val);
	return r;
}