#pragma once

//Implementation and interface copied from Mitsuba.
#include <vector>
#include "Texture.h"
#include "Samples.h"
#include "PhaseFunction.h"
#include <Engine/MicrofacetDistribution.h>
#include <Math/MonteCarlo.h>
#include <Math/Warp.h>
#include <Math/FresnelHelper.h>
#include <Base/VirtualFuncType.h>
#include "Dispersion.h"

namespace CudaTracerLib
{
#define NUM_TEX_PER_BSDF 10

struct BSDF : public BaseType//, public BaseTypeHelper<4608834>
{
	unsigned int m_combinedType;
	unsigned int m_uTextureOffsets[NUM_TEX_PER_BSDF];
	bool m_enableTwoSided;
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
	CUDA_FUNC_IN unsigned int getType() const
	{
		return m_combinedType;
	}
	CUDA_FUNC_IN bool hasComponent(unsigned int type) const {
		return (type & m_combinedType) != 0;
	}
	BSDF()
		: m_combinedType(0), m_enableTwoSided(false)
	{
	}
	BSDF(EBSDFType type)
		: m_combinedType(type), m_enableTwoSided(false)
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
	CUDA_FUNC_IN Texture* getTexture(unsigned int idx)
	{
		return (Texture*)((unsigned long long)this + m_uTextureOffsets[idx]);
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
		return sample_Helper::Caller<Spectrum>(this, bRec, pdf, _sample);
	}
	CUDA_FUNC_IN Spectrum sample(BSDFSamplingRecord &bRec, const Vec2f &_sample) const
	{
		float p;
		return sample(bRec, p, _sample);
	}
	CALLER(f)
	CUDA_FUNC_IN Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure = ESolidAngle) const
	{
		return f_Helper::Caller<Spectrum>(this, bRec, measure);
	}
	CALLER(pdf)
	CUDA_FUNC_IN float pdf(const BSDFSamplingRecord &bRec, EMeasure measure = ESolidAngle) const
	{
		return pdf_Helper::Caller<float>(this, bRec, measure);
	}
	CUDA_FUNC_IN unsigned int getType() const
	{
		return As()->getType();
	}
	CUDA_FUNC_IN bool hasComponent(unsigned int type) const {
		return As()->hasComponent(type);
	}
};

}

#include "BSDF_Complex.h"

namespace CudaTracerLib
{

struct BSDFALL : public CudaVirtualAggregate<BSDF, diffuse, roughdiffuse, dielectric, thindielectric, roughdielectric, conductor, roughconductor, plastic, roughplastic, phong, ward, hk, coating, roughcoating, blend>
{
private:
	template<bool wi, bool wo> CUDA_FUNC_IN bool start_two_sided(BSDFSamplingRecord& bRec) const
	{
		bool flip = bRec.wi.z < 0 && As()->m_enableTwoSided;
		if (flip)
		{
			if (wi)
				bRec.wi.z *= -1.0f;
			if(wo)
				bRec.wo.z *= -1.0f;
		}
		return flip;
	}
	template<bool wi, bool wo> CUDA_FUNC_IN void end_two_sided(BSDFSamplingRecord& bRec, bool flip) const
	{
		if (flip)
		{
			if (wi)
				bRec.wi.z *= -1.0f;
			if (wo)
				bRec.wo.z *= -1.0f;
		}
	}
public:
	BSDFALL()
	{

	}
	CALLER(sample)
	CUDA_FUNC_IN Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const Vec2f &_sample) const
	{
		bool flip = start_two_sided<true, false>(bRec);
		auto res = sample_Helper::Caller<Spectrum>(this, bRec, pdf, _sample);
		end_two_sided<true, true>(bRec, flip);
		return res;
	}
	CUDA_FUNC_IN Spectrum sample(BSDFSamplingRecord &bRec, const Vec2f &_sample) const
	{
		float pdf;
		return sample(bRec, pdf, _sample);
	}
	CALLER(f)
	CUDA_FUNC_IN Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure = ESolidAngle) const
	{
		bool flip = start_two_sided<true, false>((BSDFSamplingRecord&)bRec);
		auto res = f_Helper::Caller<Spectrum>(this, bRec, measure);
		end_two_sided<true, true>((BSDFSamplingRecord&)bRec, flip);
		return res;
	}
	CALLER(pdf)
	CUDA_FUNC_IN float pdf(const BSDFSamplingRecord &bRec, EMeasure measure = ESolidAngle) const
	{
		bool flip = start_two_sided<true, false>((BSDFSamplingRecord&)bRec);
		float res = pdf_Helper::Caller<float>(this, bRec, measure);
		end_two_sided<true, true>((BSDFSamplingRecord&)bRec, flip);
		return res;
	}
	CUDA_FUNC_IN unsigned int getType() const
	{
		return As()->getType();
	}
	CUDA_FUNC_IN bool hasComponent(unsigned int type) const {
		return As()->hasComponent(type);
	}
};

}
