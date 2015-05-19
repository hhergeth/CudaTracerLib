#pragma once

#include "../MathTypes.h"

struct PhaseFunctionSamplingRecord;
struct CudaRNG;

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

struct e_BasePhaseFunction : public e_BaseType//, public e_BaseTypeHelper<4408912>
{
	EPhaseFunctionType type;

	e_BasePhaseFunction(){}
	e_BasePhaseFunction(EPhaseFunctionType t)
		: type(t)
	{

	}
};

struct e_HGPhaseFunction : public e_BasePhaseFunction//, public e_DerivedTypeHelper<1>
{
	TYPE_FUNC(1)
	float m_g;

	e_HGPhaseFunction(){}
	e_HGPhaseFunction(float g)
		: e_BasePhaseFunction(EPhaseFunctionType::pEAngleDependence), m_g(g)
	{
	}

	CUDA_DEVICE CUDA_HOST float Evaluate(const PhaseFunctionSamplingRecord &pRec) const;

	CUDA_DEVICE CUDA_HOST float Sample(PhaseFunctionSamplingRecord &pRec, CudaRNG& sampler) const;

	CUDA_DEVICE CUDA_HOST float Sample(PhaseFunctionSamplingRecord &pRec, float &pdf, CudaRNG& sampler) const;
};

struct e_IsotropicPhaseFunction : public e_BasePhaseFunction//, public e_DerivedTypeHelper<2>
{
	TYPE_FUNC(2)
	e_IsotropicPhaseFunction()
		: e_BasePhaseFunction((EPhaseFunctionType)(pEIsotropic | pEAngleDependence))
	{
	}

	CUDA_DEVICE CUDA_HOST float Evaluate(const PhaseFunctionSamplingRecord &pRec) const;

	CUDA_DEVICE CUDA_HOST float Sample(PhaseFunctionSamplingRecord &pRec, CudaRNG& sampler) const;

	CUDA_DEVICE CUDA_HOST float Sample(PhaseFunctionSamplingRecord &pRec, float &pdf, CudaRNG& sampler) const;
};

struct e_KajiyaKayPhaseFunction : public e_BasePhaseFunction//, public e_DerivedTypeHelper<3>
{
	TYPE_FUNC(3)
	float m_ks, m_kd, m_exponent, m_normalization;
	Vec3f orientation;
	
	e_KajiyaKayPhaseFunction(){}
	e_KajiyaKayPhaseFunction(float ks, float kd, float e, Vec3f o);

	virtual void Update();

	CUDA_DEVICE CUDA_HOST float Evaluate(const PhaseFunctionSamplingRecord &pRec) const;

	CUDA_DEVICE CUDA_HOST float Sample(PhaseFunctionSamplingRecord &pRec, CudaRNG& sampler) const;

	CUDA_DEVICE CUDA_HOST float Sample(PhaseFunctionSamplingRecord &pRec, float &pdf, CudaRNG& sampler) const;
};

struct e_RayleighPhaseFunction : public e_BasePhaseFunction//, public e_DerivedTypeHelper<4>
{
	TYPE_FUNC(4)
	e_RayleighPhaseFunction()
		: e_BasePhaseFunction(EPhaseFunctionType::pEIsotropic)
	{
	}

	CUDA_DEVICE CUDA_HOST float Evaluate(const PhaseFunctionSamplingRecord &pRec) const;

	CUDA_DEVICE CUDA_HOST float Sample(PhaseFunctionSamplingRecord &pRec, CudaRNG& sampler) const;

	CUDA_DEVICE CUDA_HOST float Sample(PhaseFunctionSamplingRecord &pRec, float &pdf, CudaRNG& sampler) const;
};

struct CUDA_ALIGN(16) e_PhaseFunction : public CudaVirtualAggregate<e_BasePhaseFunction, e_HGPhaseFunction, e_IsotropicPhaseFunction, e_KajiyaKayPhaseFunction, e_RayleighPhaseFunction>
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