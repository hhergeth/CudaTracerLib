#pragma once

#include <VirtualFuncType.h>

//Implementation and interface copied from Mitsuba as well as PBRT.

namespace CudaTracerLib {

struct PhaseFunctionSamplingRecord;
struct CudaRNG;

//this architecture and the implementations are completly copied from mitsuba!

enum EPhaseFunctionType
{
	/// Completely isotropic 1/(4 pi) phase function
	pEIsotropic = 0x01,
	/// The phase function only depends on \c dot(wi,wo)
	pEAngleDependence = 0x04,
	/// The opposite of \ref EAngleDependence (there is an arbitrary dependence)
	pEAnisotropic = 0x02,
	/// The phase function is non symmetric, i.e. eval(wi,wo) != eval(wo, wi)
	pENonSymmetric = 0x08
};

struct BasePhaseFunction : public BaseType//, public BaseTypeHelper<4408912>
{
	EPhaseFunctionType type;

	CUDA_FUNC_IN BasePhaseFunction(EPhaseFunctionType t)
		: type(t)
	{

	}
};

struct HGPhaseFunction : public BasePhaseFunction//, public e_DerivedTypeHelper<1>
{
	TYPE_FUNC(1)
	float m_g;

	HGPhaseFunction(float g)
		: BasePhaseFunction(EPhaseFunctionType::pEAngleDependence), m_g(g)
	{
	}

	HGPhaseFunction()
		: BasePhaseFunction(EPhaseFunctionType::pEAngleDependence), m_g(0)
	{
		
	}

	CUDA_DEVICE CUDA_HOST float Evaluate(const PhaseFunctionSamplingRecord &pRec) const;

	CUDA_DEVICE CUDA_HOST float Sample(PhaseFunctionSamplingRecord &pRec, CudaRNG& sampler) const;

	CUDA_DEVICE CUDA_HOST float Sample(PhaseFunctionSamplingRecord &pRec, float &pdf, CudaRNG& sampler) const;
};

struct IsotropicPhaseFunction : public BasePhaseFunction//, public e_DerivedTypeHelper<2>
{
	TYPE_FUNC(2)

	CUDA_FUNC_IN IsotropicPhaseFunction()
		: BasePhaseFunction((EPhaseFunctionType)(pEIsotropic | pEAngleDependence))
	{
	}

	CUDA_DEVICE CUDA_HOST float Evaluate(const PhaseFunctionSamplingRecord &pRec) const;

	CUDA_DEVICE CUDA_HOST float Sample(PhaseFunctionSamplingRecord &pRec, CudaRNG& sampler) const;

	CUDA_DEVICE CUDA_HOST float Sample(PhaseFunctionSamplingRecord &pRec, float &pdf, CudaRNG& sampler) const;
};

struct KajiyaKayPhaseFunction : public BasePhaseFunction//, public e_DerivedTypeHelper<3>
{
	TYPE_FUNC(3)
	float m_ks, m_kd, m_exponent, m_normalization;

	KajiyaKayPhaseFunction();

	KajiyaKayPhaseFunction(float ks, float kd, float e);

	virtual void Update();

	CUDA_DEVICE CUDA_HOST float Evaluate(const PhaseFunctionSamplingRecord &pRec) const;

	CUDA_DEVICE CUDA_HOST float Sample(PhaseFunctionSamplingRecord &pRec, CudaRNG& sampler) const;

	CUDA_DEVICE CUDA_HOST float Sample(PhaseFunctionSamplingRecord &pRec, float &pdf, CudaRNG& sampler) const;
};

struct RayleighPhaseFunction : public BasePhaseFunction//, public e_DerivedTypeHelper<4>
{
	TYPE_FUNC(4)

	RayleighPhaseFunction()
		: BasePhaseFunction(EPhaseFunctionType::pEIsotropic)
	{
	}

	CUDA_DEVICE CUDA_HOST float Evaluate(const PhaseFunctionSamplingRecord &pRec) const;

	CUDA_DEVICE CUDA_HOST float Sample(PhaseFunctionSamplingRecord &pRec, CudaRNG& sampler) const;

	CUDA_DEVICE CUDA_HOST float Sample(PhaseFunctionSamplingRecord &pRec, float &pdf, CudaRNG& sampler) const;
};

struct PhaseFunction : public CudaVirtualAggregate<BasePhaseFunction, HGPhaseFunction, IsotropicPhaseFunction, KajiyaKayPhaseFunction, RayleighPhaseFunction>
{
public:
	CUDA_FUNC_IN EPhaseFunctionType getType() const
	{
		return As()->type;
	}

	CALLER(Evaluate)
	CUDA_FUNC_IN float Evaluate(const PhaseFunctionSamplingRecord &pRec) const
	{
		return Evaluate_Helper::Caller<float>(this, pRec);
	}

	CALLER(Sample)
	CUDA_FUNC_IN float Sample(PhaseFunctionSamplingRecord &pRec, CudaRNG& sampler) const
	{
		return Sample_Helper::Caller<float>(this, pRec, sampler);
	}

	CUDA_FUNC_IN float Sample(PhaseFunctionSamplingRecord &pRec, float &pdf, CudaRNG& sampler) const
	{
		return Sample_Helper::Caller<float>(this, pRec, pdf, sampler);
	}

	CALLER(pdf)
	CUDA_FUNC_IN float pdf(const PhaseFunctionSamplingRecord &pRec) const
	{
		return Evaluate(pRec);
	}
};

}