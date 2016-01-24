#pragma once

#include <MathTypes.h>
#include "ShapeSet.h"
#include "MIPMap_device.h"
#include "AbstractEmitter.h"
#include "Samples.h"
#include <VirtualFuncType.h>
#include "Texture.h"

//Implementation and interface copied from Mitsuba.

namespace CudaTracerLib {

template<typename H, typename D> class BufferReference;
template<typename T> class Stream;
struct KernelMIPMap;
class MIPMap;
struct TriIntersectorData;

struct LightBase : public AbstractEmitter//, public BaseTypeHelper<5523276>
{
	LightBase()
		: AbstractEmitter(0)
	{

	}

	LightBase(unsigned int type)
		: AbstractEmitter(type)
	{
	}
};

struct PointLight : public LightBase//, public e_DerivedTypeHelper<1>
{
	TYPE_FUNC(1)
	Vec3f lightPos;
	Spectrum m_intensity;

	PointLight()
		: LightBase(EDeltaPosition)
	{
		
	}

	PointLight(Vec3f p, Spectrum L, float r = 0)
		: LightBase(EDeltaPosition), lightPos(p), m_intensity(L)
	{

	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleRay(Ray &ray, const Vec2f &spatialSample, const Vec2f &directionalSample) const;

	CUDA_FUNC_IN Spectrum eval(const Vec3f& p, const Frame& sys, const NormalizedT<Vec3f> &d) const
	{
		return Spectrum(0.0f);
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const;

	CUDA_FUNC_IN float pdfDirect(const DirectSamplingRecord &dRec) const
	{
		return dRec.measure == EDiscrete ? 1.0f : 0.0f;
	}

	CUDA_DEVICE CUDA_HOST Spectrum samplePosition(PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const;

	CUDA_FUNC_IN Spectrum evalPosition(const PositionSamplingRecord &pRec) const
	{
		return (pRec.measure == EDiscrete) ? (m_intensity * 4 * PI) : Spectrum(0.0f);
	}

	CUDA_FUNC_IN float pdfPosition(const PositionSamplingRecord &pRec) const
	{
		return (pRec.measure == EDiscrete) ? 1.0f : 0.0f;
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const;

	CUDA_FUNC_IN float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		return (dRec.measure == ESolidAngle) ? INV_FOURPI : 0.0f;
	}

	CUDA_FUNC_IN Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		return Spectrum((dRec.measure == ESolidAngle) ? INV_FOURPI : 0.0f);
	}

	CUDA_FUNC_IN AABB getBox(float eps) const
	{
		return AABB(lightPos - Vec3f(eps), lightPos + Vec3f(eps));
	}
};

struct DiffuseLight : public LightBase//, public e_DerivedTypeHelper<2>
{
	TYPE_FUNC(2)
	CUDA_ALIGN(16) ShapeSet shapeSet;
	CUDA_ALIGN(16) Texture m_rad_texture;
	bool m_bOrthogonal;
	unsigned int m_uNodeIdx;

	DiffuseLight()
		: LightBase(EOnSurface)
	{
		
	}

	DiffuseLight(const Spectrum& L, ShapeSet& s, unsigned int nodeIdx)
		: LightBase(EOnSurface), shapeSet(s), m_bOrthogonal(false), m_uNodeIdx(nodeIdx)
	{
		m_rad_texture.SetData(ConstantTexture(L));
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleRay(Ray &ray, const Vec2f &spatialSample, const Vec2f &directionalSample) const;

	CUDA_DEVICE CUDA_HOST Spectrum eval(const Vec3f& p, const Frame& sys, const NormalizedT<Vec3f> &d) const;

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const;

	CUDA_DEVICE CUDA_HOST float pdfDirect(const DirectSamplingRecord &dRec) const;

	CUDA_DEVICE CUDA_HOST Spectrum samplePosition(PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const;

	CUDA_DEVICE CUDA_HOST Spectrum evalPosition(const PositionSamplingRecord &pRec) const;

	CUDA_FUNC_IN float pdfPosition(const PositionSamplingRecord &pRec) const
	{
		return shapeSet.PdfPosition();
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const;

	CUDA_DEVICE CUDA_HOST float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const;

	CUDA_DEVICE CUDA_HOST Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const;

	CUDA_FUNC_IN AABB getBox(float eps) const
	{
		return shapeSet.getBox();
	}
};

struct DistantLight : public LightBase//, public e_DerivedTypeHelper<3>
{
	TYPE_FUNC(3)
	Spectrum m_normalIrradiance, m_power;
	Frame ToWorld;
	float m_invSurfaceArea, radius;

	DistantLight()
		: LightBase(EDeltaDirection)
	{
		
	}

	///r is the radius of the scene's bounding sphere
	DistantLight(const Spectrum& L, NormalizedT<Vec3f> d, float r)
		: LightBase(EDeltaDirection), ToWorld(d), radius(r * 1.1f)
	{
		float surfaceArea = PI * radius * radius;
		m_invSurfaceArea = 1.0f / surfaceArea;
		setEmit(L);
	}

	virtual void Update()
	{
		ToWorld = Frame(ToWorld.n);
		float surfaceArea = PI * radius * radius;
		m_invSurfaceArea = 1.0f / surfaceArea;
		setEmit(m_normalIrradiance);
	}

	void setEmit(const Spectrum& L);

	void scaleEmit(const Spectrum& L)
	{
		setEmit(m_normalIrradiance * L);
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleRay(Ray &ray, const Vec2f &spatialSample, const Vec2f &directionalSample) const;

	CUDA_FUNC_IN Spectrum eval(const Vec3f& p, const Frame& sys, const NormalizedT<Vec3f> &d) const
	{
		return Spectrum(0.0f);
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const;

	CUDA_FUNC_IN float pdfDirect(const DirectSamplingRecord &dRec) const
	{
		return dRec.measure == EDiscrete ? 1.0f : 0.0f;
	}

	CUDA_DEVICE CUDA_HOST Spectrum samplePosition(PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const;

	CUDA_FUNC_IN Spectrum evalPosition(const PositionSamplingRecord &pRec) const
	{
		return (pRec.measure == EArea) ? m_normalIrradiance : Spectrum(0.0f);
	}

	CUDA_FUNC_IN float pdfPosition(const PositionSamplingRecord &pRec) const
	{
		return (pRec.measure == EArea) ? m_invSurfaceArea : 0.0f;
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const;

	CUDA_FUNC_IN float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		return (dRec.measure == EDiscrete) ? 1.0f : 0.0f;
	}

	CUDA_FUNC_IN Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		return Spectrum((dRec.measure == EDiscrete) ? 1.0f : 0.0f);
	}

	CUDA_FUNC_IN AABB getBox(float eps) const
	{
		return AABB(Vec3f(-radius), Vec3f(+radius));
	}
};

struct SpotLight : public LightBase//, public e_DerivedTypeHelper<4>
{
	TYPE_FUNC(4)
	Spectrum m_intensity;
	float m_beamWidth, m_cutoffAngle;
	float m_cosBeamWidth, m_cosCutoffAngle, m_invTransitionWidth;
	Frame ToWorld;
	Vec3f Position, Target;

	SpotLight()
		: LightBase(EDeltaPosition)
	{
		
	}

	SpotLight(Vec3f p, Vec3f t, Spectrum L, float width, float fall);

	virtual void Update()
	{
		ToWorld = Frame((Target - Position).normalized());
		m_cosBeamWidth = cosf(m_beamWidth);
		m_cosCutoffAngle = cosf(m_cutoffAngle);
		m_invTransitionWidth = 1.0f / (m_cutoffAngle - m_beamWidth);
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleRay(Ray &ray, const Vec2f &spatialSample, const Vec2f &directionalSample) const;

	CUDA_FUNC_IN Spectrum eval(const Vec3f& p, const Frame& sys, const NormalizedT<Vec3f> &d) const
	{
		return Spectrum(0.0f);
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const;

	CUDA_FUNC_IN float pdfDirect(const DirectSamplingRecord &dRec) const
	{
		return dRec.measure == EDiscrete ? 1.0f : 0.0f;
	}

	CUDA_DEVICE CUDA_HOST Spectrum samplePosition(PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const;

	CUDA_FUNC_IN Spectrum evalPosition(const PositionSamplingRecord &pRec) const
	{
		return (pRec.measure == EDiscrete) ? (m_intensity * 4 * PI) : Spectrum(0.0f);
	}

	CUDA_FUNC_IN float pdfPosition(const PositionSamplingRecord &pRec) const
	{
		return (pRec.measure == EDiscrete) ? 1.0f : 0.0f;
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const;

	CUDA_FUNC_IN float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		return (dRec.measure == ESolidAngle) ? Warp::squareToUniformConePdf(m_cosCutoffAngle) : 0.0f;
	}

	CUDA_FUNC_IN Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		return (dRec.measure == ESolidAngle) ? falloffCurve(ToWorld.toLocal(dRec.d)) * INV_FOURPI : Spectrum(0.0f);
	}

	CUDA_FUNC_IN AABB getBox(float eps) const
	{
		return AABB(Position - Vec3f(eps), Position + Vec3f(eps));
	}
private:
	CUDA_DEVICE CUDA_HOST Spectrum falloffCurve(const NormalizedT<Vec3f> &d) const;
};

struct InfiniteLight : public LightBase//, public e_DerivedTypeHelper<5>
{
	TYPE_FUNC(5)
	KernelMIPMap radianceMap;
	e_Variable<float> m_cdfRows, m_cdfCols, m_rowWeights;
	Vec3f m_SceneCenter;
	float m_SceneRadius;
	float m_normalization;
	float m_power;
	float m_invSurfaceArea;
	Vec2f m_size, m_pixelSize;
	Spectrum m_scale;
	float4x4 m_worldTransform, m_worldTransformInverse;

	const AABB* m_pSceneBox;

	CUDA_FUNC_IN InfiniteLight() {}

	CUDA_HOST InfiniteLight(Stream<char>* a_Buffer, BufferReference<MIPMap, KernelMIPMap>& mip, const Spectrum& scale, const AABB* scenBox);

	virtual void Update()
	{
		m_SceneCenter = m_pSceneBox->Center();
		m_SceneRadius = m_pSceneBox->Size().length() / 1.5f;
		float surfaceArea = 4 * PI * m_SceneRadius * m_SceneRadius;
		m_invSurfaceArea = 1 / surfaceArea;
		m_power = (surfaceArea * m_scale / m_normalization).average();
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleRay(Ray &ray, const Vec2f &spatialSample, const Vec2f &directionalSample) const;

	CUDA_FUNC_IN Spectrum eval(const Vec3f& p, const Frame& sys, const NormalizedT<Vec3f> &d) const
	{
		return evalEnvironment(Ray(p, -1.0f * d));
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const;

	CUDA_DEVICE CUDA_HOST float pdfDirect(const DirectSamplingRecord &dRec) const;

	CUDA_DEVICE CUDA_HOST Spectrum samplePosition(PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const;

	CUDA_FUNC_IN Spectrum evalPosition(const PositionSamplingRecord &pRec) const
	{
		return Spectrum(m_power * m_invSurfaceArea);
	}

	CUDA_FUNC_IN float pdfPosition(const PositionSamplingRecord &pRec) const
	{
		return m_invSurfaceArea;
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const;

	CUDA_FUNC_IN float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		return internalPdfDirection(m_worldTransformInverse.TransformDirection(-dRec.d));
	}

	CUDA_DEVICE CUDA_HOST Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const;

	CUDA_DEVICE CUDA_HOST Spectrum evalEnvironment(const Ray &ray) const;

	CUDA_DEVICE CUDA_HOST Spectrum evalEnvironment(const Ray &ray, const Ray& rX, const Ray& rY) const;

	CUDA_FUNC_IN AABB getBox(float eps) const
	{
		return AABB(-Vec3f(1.0f / eps), Vec3f(1.0f / eps));
	}
private:
	CUDA_DEVICE CUDA_HOST void internalSampleDirection(Vec2f sample, Vec3f &d, Spectrum &value, float &pdf) const;
	CUDA_DEVICE CUDA_HOST float internalPdfDirection(const Vec3f &d) const;
};

struct Light : public CudaVirtualAggregate<LightBase, PointLight, DiffuseLight, DistantLight, SpotLight, InfiniteLight>
{
public:
	CALLER(sampleRay)
	CUDA_FUNC_IN Spectrum sampleRay(Ray &ray, const Vec2f &spatialSample, const Vec2f &directionalSample) const
	{
		return sampleRay_Helper::Caller<Spectrum>(this, ray, spatialSample, directionalSample);
	}

	CALLER(eval)
	CUDA_FUNC_IN Spectrum eval(const Vec3f& p, const Frame& sys, const NormalizedT<Vec3f> &d) const
	{
		return eval_Helper::Caller<Spectrum>(this, p, sys, d);
	}

	CALLER(sampleDirect)
	CUDA_FUNC_IN Spectrum sampleDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const
	{
		return sampleDirect_Helper::Caller<Spectrum>(this, dRec, sample);
	}

	CALLER(pdfDirect)
	CUDA_FUNC_IN float pdfDirect(const DirectSamplingRecord &dRec) const
	{
		return pdfDirect_Helper::Caller<float>(this, dRec);
	}

	CALLER(samplePosition)
	CUDA_FUNC_IN Spectrum samplePosition(PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra = 0) const
	{
		return samplePosition_Helper::Caller<Spectrum>(this, pRec, sample, extra);
	}

	CALLER(evalPosition)
	CUDA_FUNC_IN Spectrum evalPosition(const PositionSamplingRecord &pRec) const
	{
		return evalPosition_Helper::Caller<Spectrum>(this, pRec);
	}

	CALLER(pdfPosition)
	CUDA_FUNC_IN float pdfPosition(const PositionSamplingRecord &pRec) const
	{
		return pdfPosition_Helper::Caller<float>(this, pRec);
	}

	CALLER(sampleDirection)
	CUDA_FUNC_IN Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra = 0) const
	{
		return sampleDirection_Helper::Caller<Spectrum>(this, dRec, pRec, sample, extra);
	}

	CALLER(pdfDirection)
	CUDA_FUNC_IN float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		return pdfDirection_Helper::Caller<float>(this, dRec, pRec);
	}

	CALLER(evalDirection)
	CUDA_FUNC_IN Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		return evalDirection_Helper::Caller<Spectrum>(this, dRec, pRec);
	}

	CALLER(getBox)
	CUDA_FUNC_IN AABB getBox(float eps) const
	{
		return getBox_Helper::Caller<AABB>(this, eps);
	}
};

}
