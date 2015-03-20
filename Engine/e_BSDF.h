#pragma once

//this architecture and the implementations are completly copied from mitsuba!

#include <MathTypes.h>
#include "Engine\e_KernelTexture.h"
#include "Engine/e_Samples.h"
#include "Engine\e_PhaseFunction.h"
#include "e_MicrofacetDistribution.h"
#include "e_RoughTransmittance.h"

#define NUM_TEX_PER_BSDF 10

struct BSDF : public e_BaseType//, public e_BaseTypeHelper<4608834>
{
	unsigned int m_combinedType;
	unsigned int m_uTextureOffsets[NUM_TEX_PER_BSDF];
	void initTextureOffsets(std::vector<e_Texture*>* nestedTexs = 0,
		e_Texture* tex0 = 0, e_Texture* tex1 = 0, e_Texture* tex2 = 0, e_Texture* tex3 = 0, e_Texture* tex4 = 0,
		e_Texture* tex5 = 0, e_Texture* tex6 = 0, e_Texture* tex7 = 0, e_Texture* tex8 = 0, e_Texture* tex9 = 0)
	{
		m_uTextureOffsets[0] = tex0 ? unsigned int((unsigned long long)tex0 - (unsigned long long)this) : 0;
		m_uTextureOffsets[1] = tex1 ? unsigned int((unsigned long long)tex1 - (unsigned long long)this) : 0;
		m_uTextureOffsets[2] = tex2 ? unsigned int((unsigned long long)tex2 - (unsigned long long)this) : 0;
		m_uTextureOffsets[3] = tex3 ? unsigned int((unsigned long long)tex3 - (unsigned long long)this) : 0;
		m_uTextureOffsets[4] = tex4 ? unsigned int((unsigned long long)tex4 - (unsigned long long)this) : 0;
		m_uTextureOffsets[5] = tex5 ? unsigned int((unsigned long long)tex5 - (unsigned long long)this) : 0;
		m_uTextureOffsets[6] = tex6 ? unsigned int((unsigned long long)tex6 - (unsigned long long)this) : 0;
		m_uTextureOffsets[7] = tex7 ? unsigned int((unsigned long long)tex7 - (unsigned long long)this) : 0;
		m_uTextureOffsets[8] = tex8 ? unsigned int((unsigned long long)tex8 - (unsigned long long)this) : 0;
		m_uTextureOffsets[9] = tex9 ? unsigned int((unsigned long long)tex9 - (unsigned long long)this) : 0;
		int n = 0;
		while (n < NUM_TEX_PER_BSDF && m_uTextureOffsets[n] != 0)
			n++;
		if (nestedTexs)
		{
			if (n + nestedTexs->size() > NUM_TEX_PER_BSDF)
				throw std::runtime_error("Too many textures in bsdf!");
			for (size_t i = 0; i < nestedTexs->size(); i++)
				m_uTextureOffsets[n++] = unsigned int((unsigned long long)nestedTexs->operator[](i) - (unsigned long long)this);
			delete nestedTexs;
		}
	}
	CUDA_FUNC_IN unsigned int getType()
	{
		return m_combinedType;
	}
	CUDA_FUNC_IN bool hasComponent(unsigned int type) const {
		return (type & m_combinedType) != 0;
	}
	BSDF(std::vector<e_Texture*>* nestedTexs = 0, 
					  e_Texture* tex0 = 0, e_Texture* tex1 = 0, e_Texture* tex2 = 0, e_Texture* tex3 = 0, e_Texture* tex4 = 0,
					  e_Texture* tex5 = 0, e_Texture* tex6 = 0, e_Texture* tex7 = 0, e_Texture* tex8 = 0, e_Texture* tex9 = 0)
		: m_combinedType(0)
	{
		initTextureOffsets(nestedTexs, tex0, tex1, tex2, tex3, tex4, tex5, tex6, tex7, tex8, tex9);
	}
	BSDF(EBSDFType type, std::vector<e_Texture*>* nestedTexs = 0,
					  e_Texture* tex0 = 0, e_Texture* tex1 = 0, e_Texture* tex2 = 0, e_Texture* tex3 = 0, e_Texture* tex4 = 0,
					  e_Texture* tex5 = 0, e_Texture* tex6 = 0, e_Texture* tex7 = 0, e_Texture* tex8 = 0, e_Texture* tex9 = 0)
		: m_combinedType(type)
	{
		initTextureOffsets(nestedTexs, tex0, tex1, tex2, tex3, tex4, tex5, tex6, tex7, tex8, tex9);
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
	std::vector<e_Texture*> getTextureList()
	{
		std::vector<e_Texture*> texs;
		int n = 0;
		while (n < NUM_TEX_PER_BSDF && m_uTextureOffsets[n] != 0)
			texs.push_back((e_Texture*)((unsigned long long)this + m_uTextureOffsets[n++]));
		return texs;
	}
	template<typename T> void LoadTextures(T clb)
	{
		std::vector<e_Texture*> T = getTextureList();
		for (size_t i = 0; i < T.size(); i++)
			T[i]->LoadTextures(clb);
		Update();
	}
};

#include "e_BSDF_Simple.h"

struct BSDFFirst : public CudaVirtualAggregate<BSDF, diffuse, roughdiffuse, dielectric, thindielectric, roughdielectric, conductor, roughconductor, plastic, phong, ward, hk>
{
public:
	CUDA_FUNC_IN Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const Vec2f &_sample) const
	{
		CALL_FUNC12(diffuse,roughdiffuse,dielectric,thindielectric,roughdielectric,conductor,roughconductor,plastic,roughplastic,phong,ward,hk, sample(bRec, pdf, _sample));
		return Spectrum(0.0f);
	}
	CUDA_FUNC_IN Spectrum sample(BSDFSamplingRecord &bRec, const Vec2f &_sample) const
	{
		float p;
		return sample(bRec, p, _sample);
	}
	CUDA_FUNC_IN Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure = ESolidAngle) const
	{
		CALL_FUNC12(diffuse,roughdiffuse,dielectric,thindielectric,roughdielectric,conductor,roughconductor,plastic,roughplastic,phong,ward,hk, f(bRec, measure));
		return Spectrum(0.0f);
	}
	CUDA_FUNC_IN float pdf(const BSDFSamplingRecord &bRec, EMeasure measure = ESolidAngle) const
	{
		CALL_FUNC12(diffuse,roughdiffuse,dielectric,thindielectric,roughdielectric,conductor,roughconductor,plastic,roughplastic,phong,ward,hk, pdf(bRec, measure));
		return 0.0f;
	}
	template<typename T> void LoadTextures(T callback) const
	{
		As()->LoadTextures(callback);
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
struct BSDFALL : public CudaVirtualAggregate<BSDF, diffuse, roughdiffuse, dielectric, thindielectric, roughdielectric, conductor, roughconductor, plastic, phong, ward, hk, coating, roughcoating, blend>
{
public:
	BSDFALL()
	{

	}
	CUDA_FUNC_IN Spectrum sample(BSDFSamplingRecord &bRec, float &pdf, const Vec2f &_sample) const
	{
		CALL_FUNC15(diffuse, roughdiffuse, dielectric, thindielectric, roughdielectric, conductor, roughconductor, plastic, roughplastic, phong, ward, hk, coating, roughcoating, blend, sample(bRec, pdf, _sample));
		return Spectrum(0.0f);
	}
	CUDA_FUNC_IN Spectrum sample(BSDFSamplingRecord &bRec, const Vec2f &_sample) const
	{
		float pdf;
		return sample(bRec, pdf, _sample);
	}
	CUDA_FUNC_IN Spectrum f(const BSDFSamplingRecord &bRec, EMeasure measure = ESolidAngle) const
	{
		CALL_FUNC15(diffuse, roughdiffuse, dielectric, thindielectric, roughdielectric, conductor, roughconductor, plastic, roughplastic, phong, ward, hk, coating, roughcoating, blend, f(bRec, measure));
		return Spectrum(0.0f);
	}
	CUDA_FUNC_IN float pdf(const BSDFSamplingRecord &bRec, EMeasure measure = ESolidAngle) const
	{
		CALL_FUNC15(diffuse, roughdiffuse, dielectric, thindielectric, roughdielectric, conductor, roughconductor, plastic, roughplastic, phong, ward, hk, coating, roughcoating, blend, pdf(bRec, measure));
		return 0.0f;
	}
	template<typename T> void LoadTextures(T callback) const
	{
		As()->LoadTextures(callback);
	}
	CUDA_FUNC_IN unsigned int getType() const
	{
		return ((BSDF*)Data)->getType();
	}
	CUDA_FUNC_IN bool hasComponent(unsigned int type) const {
		return ((BSDF*)Data)->hasComponent(type);
	}
};