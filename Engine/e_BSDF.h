#pragma once

#include <MathTypes.h>
#include "Engine\e_KernelTexture.h"
#include "Engine/e_Samples.h"
#include "Engine\e_PhaseFunction.h"
#include "e_MicrofacetDistribution.h"
#include "e_RoughTransmittance.h"

struct BSDF
{
	unsigned int m_combinedType;
	CUDA_FUNC_IN unsigned int getType()
	{
		return m_combinedType;
	}
	CUDA_FUNC_IN bool hasComponent(unsigned int type) const {
		return (type & m_combinedType) != 0;
	}
	BSDF(unsigned int type)
		: m_combinedType(type)
	{
	}
	CUDA_FUNC_IN static EMeasure getMeasure(unsigned int componentType)
	{
		if (componentType & ESmooth) {
			return ESolidAngle;
		} else if (componentType & EDelta) {
			return EDiscrete;
		} else if (componentType & EDelta1D) {
			return ELength;
		} else {
			return ESolidAngle; // will never be reached^^
		}
	}
};

#include "e_BSDF_Simple.h"

struct BSDFFirst
{
#define CALL_TYPE(t,f,r) \
	case t##_TYPE : \
		r ((t*)Data)->f; \
		break;
#define CALL_FUNC(r,f) \
	switch (m_uType) \
	{ \
		CALL_TYPE(diffuse, f, r) \
		CALL_TYPE(roughdiffuse, f, r) \
		CALL_TYPE(dielectric, f, r) \
		CALL_TYPE(thindielectric, f, r) \
		CALL_TYPE(roughdielectric, f, r) \
		CALL_TYPE(conductor, f, r) \
		CALL_TYPE(roughconductor, f, r) \
		CALL_TYPE(plastic, f, r) \
		CALL_TYPE(roughplastic, f, r) \
		CALL_TYPE(phong, f, r) \
		CALL_TYPE(ward, f, r) \
		CALL_TYPE(hk, f, r) \
	}
private:
#define SZ DMAX2(DMAX5(sizeof(diffuse), sizeof(roughdiffuse), sizeof(dielectric), sizeof(thindielectric), sizeof(roughdielectric)), \
		   DMAX6(sizeof(conductor), sizeof(roughconductor), sizeof(plastic), sizeof(phong), sizeof(ward), sizeof(hk)))
	CUDA_ALIGN(16) unsigned char Data[SZ];
#undef SZ
	unsigned int m_uType;
public:
	CUDA_FUNC_IN Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &_sample) const
	{
		CALL_FUNC(return, sample(bRec, pdf, _sample));
	}
	CUDA_FUNC_IN Spectrum sample(BSDFSamplingRecord &bRec, const float2 &_sample) const
	{
		float p;
		return sample(bRec, p, _sample);
	}
	CUDA_FUNC_IN Spectrum f(BSDFSamplingRecord &bRec, EMeasure measure = ESolidAngle) const
	{
		CALL_FUNC(return, f(bRec, measure));
	}
	CUDA_FUNC_IN float pdf(BSDFSamplingRecord &bRec, EMeasure measure = ESolidAngle) const
	{
		CALL_FUNC(return, pdf(bRec, measure));
	}
	template<typename T> void LoadTextures(T callback) const
	{
		CALL_FUNC(, LoadTextures(callback));
	}
	CUDA_FUNC_IN unsigned int getType() const
	{
		CALL_FUNC(return, getType());
	}
	CUDA_FUNC_IN bool hasComponent(unsigned int type) const {
		CALL_FUNC(return, hasComponent(type));
	}
	template<typename T> T* As()
	{
		return (T*)Data;
	}
	template<typename T> void SetData(const T& val)
	{
		memcpy(Data, &val, sizeof(T));
		m_uType = T::TYPE();
	}
#undef CALL_FUNC
#undef CALL_TYPE
};

#include "e_BSDF_Complex.h"

struct BSDFALL
{		/*CALL_TYPE(coating, f, r) \
		CALL_TYPE(roughcoating, f, r) \
		CALL_TYPE(mixturebsdf, f, r) \
		CALL_TYPE(blend, f, r) \*/
#define CALL_TYPE(t,f,r) \
	case t##_TYPE : \
		r ((t*)Data)->f; \
		break;
#define CALL_FUNC(r,f) \
	switch (m_uType) \
	{ \
		CALL_TYPE(diffuse, f, r) \
		CALL_TYPE(roughdiffuse, f, r) \
		CALL_TYPE(dielectric, f, r) \
		CALL_TYPE(thindielectric, f, r) \
		CALL_TYPE(roughdielectric, f, r) \
		CALL_TYPE(conductor, f, r) \
		CALL_TYPE(roughconductor, f, r) \
		CALL_TYPE(plastic, f, r) \
		CALL_TYPE(roughplastic, f, r) \
		CALL_TYPE(phong, f, r) \
		CALL_TYPE(ward, f, r) \
		CALL_TYPE(hk, f, r) \
		CALL_TYPE(coating, f, r) \
		CALL_TYPE(roughcoating, f, r) \
	}
private:
#define SZ DMAX3(DMAX5(sizeof(diffuse), sizeof(roughdiffuse), sizeof(dielectric), sizeof(thindielectric), sizeof(roughdielectric)), \
				 DMAX6(sizeof(conductor), sizeof(roughconductor), sizeof(plastic), sizeof(phong), sizeof(ward), sizeof(hk)), \
				 sizeof(coating))
	CUDA_ALIGN(16) unsigned char Data[SZ];
#undef SZ
	unsigned int m_uType;
public:
	CUDA_FUNC_IN Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &_sample) const
	{
		CALL_FUNC(return, sample(bRec, pdf, _sample));
	}
	CUDA_FUNC_IN Spectrum sample(BSDFSamplingRecord &bRec, const float2 &_sample) const
	{
		float p;
		return sample(bRec, p, _sample);
	}
	CUDA_FUNC_IN Spectrum f(BSDFSamplingRecord &bRec, EMeasure measure = ESolidAngle) const
	{
		CALL_FUNC(return, f(bRec, measure));
	}
	CUDA_FUNC_IN float pdf(BSDFSamplingRecord &bRec, EMeasure measure = ESolidAngle) const
	{
		CALL_FUNC(return, pdf(bRec, measure));
	}
	template<typename T> void LoadTextures(T callback) const
	{
		CALL_FUNC(, LoadTextures(callback));
	}
	CUDA_FUNC_IN unsigned int getType()
	{
		CALL_FUNC(return, getType());
	}
	CUDA_FUNC_IN bool hasComponent(unsigned int type) const {
		CALL_FUNC(return, hasComponent(type));
	}
	template<typename T> T* As()
	{
		return (T*)Data;
	}
	template<typename T> void SetData(const T& val)
	{
		memcpy(Data, &val, sizeof(T));
		m_uType = T::TYPE();
	}
#undef CALL_FUNC
#undef CALL_TYPE
};