#pragma once

#include "..\Math\AABB.h"
#include "..\Base\CudaRandom.h"
#include "e_Samples.h"

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

struct e_BasePhaseFunction
{
	EPhaseFunctionType type;

	e_BasePhaseFunction(EPhaseFunctionType t)
		: type(t)
	{

	}
};

#define e_HGPhaseFunction_TYPE 1
struct e_HGPhaseFunction : public e_BasePhaseFunction
{
	float m_g;

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

	e_KajiyaKayPhaseFunction(float ks, float kd, float e, float3 o);

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

struct CUDA_ALIGN(16) e_PhaseFunction
{
#define CALL_TYPE(t,f,r) \
	case t##_TYPE : \
		r ((t*)Data)->f; \
		break;
#define CALL_FUNC(r,f) \
	switch(type) \
	{ \
		CALL_TYPE(e_HGPhaseFunction, f, r) \
		CALL_TYPE(e_IsotropicPhaseFunction, f, r) \
		CALL_TYPE(e_KajiyaKayPhaseFunction, f, r) \
		CALL_TYPE(e_RayleighPhaseFunction, f, r) \
	}
	CUDA_ALIGN(16) unsigned char Data[PHF_SIZE];
	unsigned int type;
public:
	e_PhaseFunction()
	{
		type = 0;
	}

	template<typename T> void SetData(const T& val)
	{
		memcpy(Data, &val, sizeof(T));
		type = T::TYPE();
	}

	CUDA_FUNC_IN EPhaseFunctionType getType() const
	{
		return ((e_BasePhaseFunction*)Data)->type;
	}

	CUDA_FUNC_IN float Evaluate(const PhaseFunctionSamplingRecord &pRec) const
	{
		CALL_FUNC(return, Evaluate(pRec))
	}

	CUDA_FUNC_IN float Sample(PhaseFunctionSamplingRecord &pRec, CudaRNG& sampler) const
	{
		CALL_FUNC(return, Sample(pRec, sampler))
	}

	CUDA_FUNC_IN float Sample(PhaseFunctionSamplingRecord &pRec, float &pdf, CudaRNG& sampler) const
	{
		CALL_FUNC(return, Sample(pRec, pdf, sampler))
	}

	CUDA_FUNC_IN float pdf(const PhaseFunctionSamplingRecord &pRec) const
	{
		return Evaluate(pRec);
	}
#undef CALL_FUNC
#undef CALL_TYPE
};

template<typename T> e_PhaseFunction CreatePhaseFunction(const T& val)
{
	e_PhaseFunction r;
	r.SetData(val);
	return r;
}