#pragma once

//this architecture and the implementations are completly copied from mitsuba!

#include <MathTypes.h>
#include "Engine\e_KernelTexture.h"
#include "Engine/e_Samples.h"
#include "Engine\e_PhaseFunction.h"
#include "e_MicrofacetDistribution.h"
#include "e_RoughTransmittance.h"

#define STD_DIFFUSE_REFLECTANCE \
	CUDA_FUNC_IN Spectrum getDiffuseReflectance(BSDFSamplingRecord &bRec) const \
	{ \
		float3 wo = bRec.wo, wi = bRec.wi; \
		bRec.typeMask = EDiffuseReflection; \
		bRec.wo = bRec.wi = make_float3(0, 0, 1); \
		Spectrum r = f(bRec, ESolidAngle) * PI; \
		bRec.wo = wo; bRec.wi = wi; \
		return r; \
	}

struct BSDF : public e_BaseType
{
	unsigned int m_combinedType;
	CUDA_FUNC_IN unsigned int getType()
	{
		return m_combinedType;
	}
	CUDA_FUNC_IN bool hasComponent(unsigned int type) const {
		return (type & m_combinedType) != 0;
	}
	CUDA_FUNC_IN BSDF(unsigned int type)
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

#define BSDFFirst_SIZE DMAX2(DMAX5(sizeof(diffuse), sizeof(roughdiffuse), sizeof(dielectric), sizeof(thindielectric), sizeof(roughdielectric)), \
		   DMAX6(sizeof(conductor), sizeof(roughconductor), sizeof(plastic), sizeof(phong), sizeof(ward), sizeof(hk)))
struct BSDFFirst : public e_AggregateBaseType<BSDF, BSDFFirst_SIZE>
{
public:
	CUDA_FUNC_IN Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &_sample) const
	{
		CALL_FUNC12(diffuse,roughdiffuse,dielectric,thindielectric,roughdielectric,conductor,roughconductor,plastic,roughplastic,phong,ward,hk, sample(bRec, pdf, _sample));
		return 0.0f;
	}
	CUDA_FUNC_IN Spectrum sample(BSDFSamplingRecord &bRec, const float2 &_sample) const
	{
		float p;
		return sample(bRec, p, _sample);
	}
	CUDA_FUNC_IN Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure = ESolidAngle) const
	{
		CALL_FUNC12(diffuse,roughdiffuse,dielectric,thindielectric,roughdielectric,conductor,roughconductor,plastic,roughplastic,phong,ward,hk, f(bRec, measure));
		return 0.0f;
	}
	CUDA_FUNC_IN float pdf(const BSDFSamplingRecord &bRec, EMeasure measure = ESolidAngle) const
	{
		CALL_FUNC12(diffuse,roughdiffuse,dielectric,thindielectric,roughdielectric,conductor,roughconductor,plastic,roughplastic,phong,ward,hk, pdf(bRec, measure));
		return 0.0f;
	}
	CUDA_FUNC_IN Spectrum getDiffuseReflectance(BSDFSamplingRecord &bRec) const
	{
		CALL_FUNC12(diffuse,roughdiffuse,dielectric,thindielectric,roughdielectric,conductor,roughconductor,plastic,roughplastic,phong,ward,hk, getDiffuseReflectance(bRec));
		return 0.0f;
	}
	template<typename T> void LoadTextures(T callback) const
	{
		CALL_FUNC12(diffuse,roughdiffuse,dielectric,thindielectric,roughdielectric,conductor,roughconductor,plastic,roughplastic,phong,ward,hk, LoadTextures(callback));
	}
	CUDA_FUNC_IN unsigned int getType() const
	{
		return ((BSDF*)Data)->getType();
	}
	CUDA_FUNC_IN bool hasComponent(unsigned int type) const {
		return ((BSDF*)Data)->hasComponent(type);
	}
};

#include "e_BSDF_Complex.h"

#define BSDFALL_SIZE DMAX3(DMAX5(sizeof(diffuse), sizeof(roughdiffuse), sizeof(dielectric), sizeof(thindielectric), sizeof(roughdielectric)), \
							DMAX6(sizeof(conductor), sizeof(roughconductor), sizeof(plastic), sizeof(phong), sizeof(ward), sizeof(hk)), \
							DMAX3(sizeof(coating), sizeof(roughcoating), sizeof(blend)))
struct BSDFALL : public e_AggregateBaseType<BSDF, BSDFALL_SIZE>
{
public:
	CUDA_FUNC_IN Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const float2 &_sample) const
	{
		CALL_FUNC15(diffuse,roughdiffuse,dielectric,thindielectric,roughdielectric,conductor,roughconductor,plastic,roughplastic,phong,ward,hk,coating,roughcoating, blend, sample(bRec, pdf, _sample));
		return 0.0f;
	}
	CUDA_FUNC_IN Spectrum sample(BSDFSamplingRecord &bRec, const float2 &_sample) const
	{
		float p;
		return sample(bRec, p, _sample);
	}
	CUDA_FUNC_IN Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure = ESolidAngle) const
	{
		CALL_FUNC15(diffuse,roughdiffuse,dielectric,thindielectric,roughdielectric,conductor,roughconductor,plastic,roughplastic,phong,ward,hk,coating,roughcoating, blend, f(bRec, measure));
		return 0.0f;
	}
	CUDA_FUNC_IN float pdf(const BSDFSamplingRecord &bRec, EMeasure measure = ESolidAngle) const
	{
		CALL_FUNC15(diffuse,roughdiffuse,dielectric,thindielectric,roughdielectric,conductor,roughconductor,plastic,roughplastic,phong,ward,hk,coating,roughcoating, blend, pdf(bRec, measure));
		return 0.0f;
	}
	CUDA_FUNC_IN Spectrum getDiffuseReflectance(BSDFSamplingRecord &bRec) const
	{
		CALL_FUNC15(diffuse,roughdiffuse,dielectric,thindielectric,roughdielectric,conductor,roughconductor,plastic,roughplastic,phong,ward,hk,coating,roughcoating, blend, getDiffuseReflectance(bRec));
		return 0.0f;
	}
	template<typename T> void LoadTextures(T callback) const
	{
		CALL_FUNC15(diffuse,roughdiffuse,dielectric,thindielectric,roughdielectric,conductor,roughconductor,plastic,roughplastic,phong,ward,hk,coating,roughcoating, blend, LoadTextures(callback));
	}
	CUDA_FUNC_IN unsigned int getType() const
	{
		return ((BSDF*)Data)->getType();
	}
	CUDA_FUNC_IN bool hasComponent(unsigned int type) const {
		return ((BSDF*)Data)->hasComponent(type);
	}
};