#pragma once

#include "..\Math\vector.h"
#include "..\Math\AABB.h"
#include "..\Base\CudaRandom.h"
#include "..\Math\Montecarlo.h"

enum ETransportMode
{
	ERadiance = 0,
	EImportance = 1,
};

struct PhaseFunctionSamplingRecord
{
	float3 wi;
	float3 wo;
	ETransportMode mode;

	CUDA_FUNC_IN PhaseFunctionSamplingRecord(const float3& _wo, ETransportMode m = ERadiance)
	{
		wo = _wo;
		mode = m;
	}

	CUDA_FUNC_IN PhaseFunctionSamplingRecord(const float3& _wo, const float3& _wi, ETransportMode m = ERadiance)
	{
		wo = _wo;
		wi = _wi;
		mode = m;
	}

	CUDA_FUNC_IN void reverse()
	{
		swapk(&wo, &wi);
		mode = (ETransportMode) (1-mode);
	}
};

enum EPhaseFunctionType
{
	/// Completely isotropic 1/(4 pi) phase function
	EIsotropic       = 0x01,
	/// The phase function only depends on \c dot(wi,wo)
	EAngleDependence = 0x04,
	/// The opposite of \ref EAngleDependence (there is an arbitrary dependence)
	EAnisotropic     = 0x02,
	/// The phase function is non symmetric, i.e. eval(wi,wo) != eval(wo, wi)
	ENonSymmetric    = 0x08
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
		: e_BasePhaseFunction(EAngleDependence), m_g(g)
	{
	}

	CUDA_FUNC_IN float Evaluate(const PhaseFunctionSamplingRecord &pRec) const
	{
		float temp = 1.0f + m_g*m_g + 2.0f * m_g * dot(pRec.wi, pRec.wo);
		return (1.0f / (4.0f * PI)) * (1 - m_g*m_g) / (temp * sqrtf(temp));
	}

	CUDA_FUNC_IN float Sample(PhaseFunctionSamplingRecord &pRec, CudaRNG& sampler) const
	{
		float2 sample = sampler.randomFloat2();

		float cosTheta;
		if (std::abs(m_g) < EPSILON)
		{
			cosTheta = 1 - 2*sample.x;
		}
		else
		{
			float sqrTerm = (1 - m_g * m_g) / (1 - m_g + 2 * m_g * sample.x);
			cosTheta = (1 + m_g * m_g - sqrTerm * sqrTerm) / (2 * m_g);
		}

		float sinTheta = sqrtf(1.0f-cosTheta*cosTheta), sinPhi, cosPhi;

		sincos(2*PI*sample.y, &sinPhi, &cosPhi);

		pRec.wo = Onb(-pRec.wi).localToworld(make_float3(
			sinTheta * cosPhi,
			sinTheta * sinPhi,
			cosTheta
		));

		return 1.0f;
	}

	CUDA_FUNC_IN float Sample(PhaseFunctionSamplingRecord &pRec, float &pdf, CudaRNG& sampler) const
	{
		e_HGPhaseFunction::Sample(pRec, sampler);
		pdf = e_HGPhaseFunction::Evaluate(pRec);
		return 1.0f;
	}

	TYPE_FUNC(e_HGPhaseFunction)
};

#define e_IsotropicPhaseFunction_TYPE 2
struct e_IsotropicPhaseFunction : public e_BasePhaseFunction
{
	e_IsotropicPhaseFunction()
		: e_BasePhaseFunction((EPhaseFunctionType)(EIsotropic | EAngleDependence))
	{
	}

	CUDA_FUNC_IN float Evaluate(const PhaseFunctionSamplingRecord &pRec) const
	{
		return Warp::squareToUniformSpherePdf();
	}

	CUDA_FUNC_IN float Sample(PhaseFunctionSamplingRecord &pRec, CudaRNG& sampler) const
	{
		float2 sample = sampler.randomFloat2();
		pRec.wo = Warp::squareToUniformSphere(sample);
		return 1.0f;
	}

	CUDA_FUNC_IN float Sample(PhaseFunctionSamplingRecord &pRec, float &pdf, CudaRNG& sampler) const
	{
		pRec.wo = Warp::squareToUniformSphere(sampler.randomFloat2());
		pdf = Warp::squareToUniformSpherePdf();
		return 1.0f;
	}

	TYPE_FUNC(e_IsotropicPhaseFunction)
};

#define e_KajiyaKayPhaseFunction_TYPE 3
struct e_KajiyaKayPhaseFunction : public e_BasePhaseFunction
{
	float m_ks, m_kd, m_exponent, m_normalization;
	float3 orientation;

	e_KajiyaKayPhaseFunction(float ks, float kd, float e, float3 o)
		: e_BasePhaseFunction(EAnisotropic), m_ks(ks), m_kd(kd), m_exponent(e), orientation(o)
	{
		int nParts = 1000;
		float stepSize = PI / nParts, m=4, theta = stepSize;

		m_normalization = 0; /* 0 at the endpoints */
		for (int i=1; i<nParts; ++i) {
			float value = std::pow(std::cos(theta - PI/2), m_exponent)
				* std::sin(theta);
			m_normalization += value * m;
			theta += stepSize;
			m = 6-m;
		}

		m_normalization = 1/(m_normalization * stepSize/3 * 2 * PI);
	}

	CUDA_FUNC_IN float Evaluate(const PhaseFunctionSamplingRecord &pRec) const
	{
		if (length(orientation) == 0)
			return m_kd / (4*PI);

		Onb frame(normalize(orientation));
		float3 reflectedLocal = frame.worldTolocal(pRec.wo);

		reflectedLocal.z = -dot(pRec.wi, frame.m_normal);
		float a = std::sqrt((1-reflectedLocal.z*reflectedLocal.z) / (reflectedLocal.x*reflectedLocal.x + reflectedLocal.y*reflectedLocal.y));
		reflectedLocal.y *= a;
		reflectedLocal.x *= a;
		float3 R = frame.localToworld(reflectedLocal);

		return powf(MAX(0.0f, dot(R, pRec.wo)), m_exponent) * m_normalization * m_ks + m_kd / (4*PI);
	}

	CUDA_FUNC_IN float Sample(PhaseFunctionSamplingRecord &pRec, CudaRNG& sampler) const
	{
		pRec.wo = Warp::squareToUniformSphere(sampler.randomFloat2());
		return Evaluate(pRec) * (4 * PI);
	}

	CUDA_FUNC_IN float Sample(PhaseFunctionSamplingRecord &pRec, float &pdf, CudaRNG& sampler) const
	{
		pRec.wo = Warp::squareToUniformSphere(sampler.randomFloat2());
		pdf = Warp::squareToUniformSpherePdf();
		return Evaluate(pRec) * (4 * PI);
	}

	TYPE_FUNC(e_KajiyaKayPhaseFunction)
};

#define e_RayleighPhaseFunction_TYPE 4
struct e_RayleighPhaseFunction : public e_BasePhaseFunction
{
	e_RayleighPhaseFunction()
		: e_BasePhaseFunction(EIsotropic)
	{
	}

	CUDA_FUNC_IN float Evaluate(const PhaseFunctionSamplingRecord &pRec) const
	{
		float mu = dot(pRec.wi, pRec.wo);
		return (3.0f/(16.0f*PI)) * (1+mu*mu);
	}

	CUDA_FUNC_IN float Sample(PhaseFunctionSamplingRecord &pRec, CudaRNG& sampler) const
	{
		float2 sample(sampler.randomFloat2());

		float z = 2 * (2*sample.x - 1),
			  tmp = std::sqrt(z*z+1),
			  A = std::pow(z+tmp, (float) (1.0f/3.0f)),
			  B = std::pow(z-tmp, (float) (1.0f/3.0f)),
			  cosTheta = A + B,
			  sinTheta = sqrtf(1.0f-cosTheta*cosTheta),
			  phi = 2*PI*sample.y, cosPhi, sinPhi;
		sincos(phi, &sinPhi, &cosPhi);

		float3 dir = make_float3(
			sinTheta * cosPhi,
			sinTheta * sinPhi,
			cosTheta);

		pRec.wo = Onb(-pRec.wi).localToworld(dir);
		return 1.0f;
	}

	CUDA_FUNC_IN float Sample(PhaseFunctionSamplingRecord &pRec, float &pdf, CudaRNG& sampler) const
	{
		e_RayleighPhaseFunction::Sample(pRec, sampler);
		pdf = e_RayleighPhaseFunction::Evaluate(pRec);
		return 1.0f;
	}

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