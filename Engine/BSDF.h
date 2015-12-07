#pragma once

//Implementation and interface copied from Mitsuba.
#include <vector>
#include <MathTypes.h>
#include "Engine/Texture.h"
#include "Engine/Samples.h"
#include "Engine/PhaseFunction.h"
#include "MicrofacetDistribution.h"
#include <Math/Sampling.h>
#include <VirtualFuncType.h>

namespace CudaTracerLib
{
#define NUM_TEX_PER_BSDF 10

struct BSDF : public BaseType//, public BaseTypeHelper<4608834>
{
	unsigned int m_combinedType;
	unsigned int m_uTextureOffsets[NUM_TEX_PER_BSDF];
private:
	template<int COUNT> void initTextureOffsets_intern(){}
	template<int COUNT, typename... Textures> void initTextureOffsets_intern(Texture& tex, Textures&&... tail)
	{
		if (COUNT >= NUM_TEX_PER_BSDF)
			throw std::runtime_error("To many textures in one BSDF!");
		m_uTextureOffsets[COUNT] = (unsigned int)((unsigned long long)&tex - (unsigned long long)this);
		initTextureOffsets_intern<COUNT + 1>(tail...);
	}
public:
	template<typename... Textures> void initTextureOffsets(Textures&&... tail)
	{
		for (int i = 0; i < NUM_TEX_PER_BSDF; i++)
			m_uTextureOffsets[i] = 0;
		initTextureOffsets_intern<0>(tail...);
	}
	template<typename... Textures> void initTextureOffsets2(const std::vector<Texture*>& nestedTexs, Textures&&... tail)
	{
		initTextureOffsets(tail...);
		int n = 0;
		while (n < NUM_TEX_PER_BSDF && m_uTextureOffsets[n] != 0)
			n++;
		if (n + nestedTexs.size() > NUM_TEX_PER_BSDF)
			throw std::runtime_error("Too many textures in bsdf!");
		for (size_t i = 0; i < nestedTexs.size(); i++)
			m_uTextureOffsets[n++] = (unsigned int)((unsigned long long)nestedTexs[i] - (unsigned long long)this);
	}
	CUDA_FUNC_IN unsigned int getType()
	{
		return m_combinedType;
	}
	CUDA_FUNC_IN bool hasComponent(unsigned int type) const {
		return (type & m_combinedType) != 0;
	}
	BSDF()
		: m_combinedType(0)
	{
	}
	BSDF(EBSDFType type)
		: m_combinedType(type)
	{
	}
	CUDA_FUNC_IN static EMeasure getMeasure(unsigned int componentType)
	{
		if (componentType & ESmooth) {
			return ESolidAngle;
		}
		else if (componentType & EDelta) {
			return EDiscrete;
		}
		else if (componentType & EDelta1D) {
			return ELength;
		}
		else {
			return ESolidAngle; // will never be reached^^
		}
	}
	std::vector<Texture*> getTextureList()
	{
		std::vector<Texture*> texs;
		int n = 0;
		while (n < NUM_TEX_PER_BSDF && m_uTextureOffsets[n] != 0)
			texs.push_back((Texture*)((unsigned long long)this + m_uTextureOffsets[n++]));
		return texs;
	}
};
}

#include "BSDF_Simple.h"

namespace CudaTracerLib
{

struct BSDFFirst : public CudaVirtualAggregate<BSDF, diffuse, roughdiffuse, dielectric, thindielectric, roughdielectric, conductor, roughconductor, plastic, roughplastic, phong, ward, hk>
{
public:
	CALLER(sample)
		CUDA_FUNC_IN Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const Vec2f &_sample) const
	{
		return sample_Caller<Spectrum>(*this, bRec, pdf, _sample);
	}
	CUDA_FUNC_IN Spectrum sample(BSDFSamplingRecord &bRec, const Vec2f &_sample) const
	{
		float p;
		return sample(bRec, p, _sample);
	}
	CALLER(f)
		CUDA_FUNC_IN Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure = ESolidAngle) const
	{
		return f_Caller<Spectrum>(*this, bRec, measure);
	}
	CALLER(pdf)
		CUDA_FUNC_IN float pdf(const BSDFSamplingRecord &bRec, EMeasure measure = ESolidAngle) const
	{
		return pdf_Caller<float>(*this, bRec, measure);
	}
	CUDA_FUNC_IN unsigned int getType() const
	{
		return ((BSDF*)Data)->getType();
	}
	CUDA_FUNC_IN bool hasComponent(unsigned int type) const {
		return ((BSDF*)Data)->hasComponent(type);
	}
};

}

#include "BSDF_Complex.h"

namespace CudaTracerLib
{

struct BSDFALL : public CudaVirtualAggregate<BSDF, diffuse, roughdiffuse, dielectric, thindielectric, roughdielectric, conductor, roughconductor, plastic, roughplastic, phong, ward, hk, coating, roughcoating, blend>
{
public:
	BSDFALL()
	{

	}
	CALLER(sample)
		CUDA_FUNC_IN Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const Vec2f &_sample) const
	{
		return sample_Caller<Spectrum>(*this, bRec, pdf, _sample);
	}
	CUDA_FUNC_IN Spectrum sample(BSDFSamplingRecord &bRec, const Vec2f &_sample) const
	{
		float pdf;
		return sample(bRec, pdf, _sample);
	}
	CALLER(f)
		CUDA_FUNC_IN Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure = ESolidAngle) const
	{
		return f_Caller<Spectrum>(*this, bRec, measure);
	}
	CALLER(pdf)
		CUDA_FUNC_IN float pdf(const BSDFSamplingRecord &bRec, EMeasure measure = ESolidAngle) const
	{
		return pdf_Caller<float>(*this, bRec, measure);
	}
	CUDA_FUNC_IN unsigned int getType() const
	{
		return ((BSDF*)Data)->getType();
	}
	CUDA_FUNC_IN bool hasComponent(unsigned int type) const {
		return ((BSDF*)Data)->hasComponent(type);
	}
};

}
