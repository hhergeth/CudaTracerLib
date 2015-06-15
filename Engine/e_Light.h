#pragma once

#include <MathTypes.h>
#include "e_ShapeSet.h"
#include "e_FileTexture_device.h"
#include "e_AbstractEmitter.h"
#include "e_Samples.h"
#include "../VirtualFuncType.h"

template<typename H, typename D> class e_BufferReference;
template<typename T> class e_Stream;
struct e_KernelMIPMap;
class e_MIPMap;

struct e_LightBase : public e_AbstractEmitter//, public e_BaseTypeHelper<5523276>
{
	bool IsRemoved;

	e_LightBase()
		: e_AbstractEmitter(0), IsRemoved(false)
	{

	}

	e_LightBase(unsigned int type)
		: e_AbstractEmitter(type), IsRemoved(false)
	{
	}
};

struct e_PointLight : public e_LightBase//, public e_DerivedTypeHelper<1>
{
	TYPE_FUNC(1)
	Vec3f lightPos;
    Spectrum m_intensity;
	
	e_PointLight()
		: e_LightBase(EDeltaPosition)
	{}
	e_PointLight(Vec3f p, Spectrum L, float r = 0)
		: e_LightBase(EDeltaPosition), lightPos(p), m_intensity(L)
	{

	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleRay(Ray &ray, const Vec2f &spatialSample, const Vec2f &directionalSample) const;

	CUDA_FUNC_IN Spectrum eval(const Vec3f& p, const Frame& sys, const Vec3f &d) const
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
		return (pRec.measure == EDiscrete) ? (m_intensity * 4*PI) : Spectrum(0.0f);
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

struct e_DiffuseLight : public e_LightBase//, public e_DerivedTypeHelper<2>
{
	TYPE_FUNC(2)
	Spectrum m_radiance, m_power;
    ShapeSet shapeSet;
	bool m_bOrthogonal;
	unsigned int m_uNodeIdx;
	
	e_DiffuseLight()
		: e_LightBase(EOnSurface)
	{}
	e_DiffuseLight(const Spectrum& L, ShapeSet& s, unsigned int nodeIdx)
		: e_LightBase(EOnSurface), shapeSet(s), m_bOrthogonal(false), m_uNodeIdx(nodeIdx)
	{
		setEmit(L);
	}

	virtual void Update()
	{
		setEmit(m_radiance);
	}

	void setEmit(const Spectrum& L);

	void scaleEmit(const Spectrum& L)
	{
		setEmit(m_radiance * L);
	}

	void Recalculate(const float4x4& mat, e_Stream<char>* buffer)
	{
		shapeSet.Recalculate(mat, buffer);
		Update();
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleRay(Ray &ray, const Vec2f &spatialSample, const Vec2f &directionalSample) const;

	CUDA_DEVICE CUDA_HOST Spectrum eval(const Vec3f& p, const Frame& sys, const Vec3f &d) const;

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const;

	CUDA_DEVICE CUDA_HOST float pdfDirect(const DirectSamplingRecord &dRec) const;

	CUDA_FUNC_IN Spectrum samplePosition(PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const
	{
		shapeSet.SamplePosition(pRec, sample);
		return m_power;
	}

	CUDA_DEVICE CUDA_HOST Spectrum evalPosition(const PositionSamplingRecord &pRec) const;

	CUDA_FUNC_IN float pdfPosition(const PositionSamplingRecord &pRec) const
	{
		return shapeSet.Pdf();
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra) const;

	CUDA_DEVICE CUDA_HOST float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const;

	CUDA_DEVICE CUDA_HOST Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const;

	CUDA_FUNC_IN AABB getBox(float eps) const
	{
		return shapeSet.getBox();
	}
};

struct e_DistantLight : public e_LightBase//, public e_DerivedTypeHelper<3>
{
	TYPE_FUNC(3)
	Spectrum m_normalIrradiance, m_power;
	Frame ToWorld;
	float m_invSurfaceArea, radius;
	
	e_DistantLight()
		: e_LightBase(EDeltaDirection)
	{}
	///r is the radius of the scene's bounding sphere
	e_DistantLight(const Spectrum& L, Vec3f d, float r)
		: e_LightBase(EDeltaDirection), ToWorld(d), radius(r * 1.1f)
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

	CUDA_FUNC_IN Spectrum eval(const Vec3f& p, const Frame& sys, const Vec3f &d) const
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

struct e_SpotLight : public e_LightBase//, public e_DerivedTypeHelper<4>
{
	TYPE_FUNC(4)
    Spectrum m_intensity;
	float m_beamWidth, m_cutoffAngle;
	float m_cosBeamWidth, m_cosCutoffAngle, m_invTransitionWidth;
	Frame ToWorld;
	Vec3f Position, Target;
	
	e_SpotLight()
		: e_LightBase(EDeltaPosition)
	{}
	e_SpotLight(Vec3f p, Vec3f t, Spectrum L, float width, float fall);

	virtual void Update()
	{
		ToWorld = Frame(Target - Position);
		m_cosBeamWidth = cosf(m_beamWidth);
		m_cosCutoffAngle = cosf(m_cutoffAngle);
		m_invTransitionWidth = 1.0f / (m_cutoffAngle - m_beamWidth);
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleRay(Ray &ray, const Vec2f &spatialSample, const Vec2f &directionalSample) const;

	CUDA_FUNC_IN Spectrum eval(const Vec3f& p, const Frame& sys, const Vec3f &d) const
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
		return (pRec.measure == EDiscrete) ? (m_intensity * 4*PI) : Spectrum(0.0f);
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
	CUDA_DEVICE CUDA_HOST Spectrum falloffCurve(const Vec3f &d) const;
};

struct e_InfiniteLight : public e_LightBase//, public e_DerivedTypeHelper<5>
{
	TYPE_FUNC(5)
	e_KernelMIPMap radianceMap;
	e_Variable<float> m_cdfRows, m_cdfCols, m_rowWeights;
	Vec3f m_SceneCenter;
	float m_SceneRadius;
	float m_normalization;
	float m_power;
	float m_invSurfaceArea;
	Vec2f m_size, m_pixelSize;
	Spectrum m_scale;
	float4x4 m_worldTransform, m_worldTransformInverse;
	
	CUDA_FUNC_IN e_InfiniteLight() {}

	CUDA_HOST e_InfiniteLight(e_Stream<char>* a_Buffer, e_BufferReference<e_MIPMap, e_KernelMIPMap>& mip, const Spectrum& scale, const AABB& scenBox);

	CUDA_DEVICE CUDA_HOST Spectrum sampleRay(Ray &ray, const Vec2f &spatialSample, const Vec2f &directionalSample) const;

	CUDA_FUNC_IN Spectrum eval(const Vec3f& p, const Frame& sys, const Vec3f &d) const
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

CUDA_ALIGN(16) struct e_KernelLight : public CudaVirtualAggregate<e_LightBase, e_PointLight, e_DiffuseLight, e_DistantLight, e_SpotLight, e_InfiniteLight>
{
public:
	CALLER(sampleRay)
	CUDA_FUNC_IN Spectrum sampleRay(Ray &ray, const Vec2f &spatialSample, const Vec2f &directionalSample) const
	{
		return sampleRay_Caller<Spectrum>(*this, ray, spatialSample, directionalSample);
	}

	CALLER(eval)
	CUDA_FUNC_IN Spectrum eval(const Vec3f& p, const Frame& sys, const Vec3f &d) const
	{
		return eval_Caller<Spectrum>(*this, p, sys, d);
	}

	CALLER(sampleDirect)
	CUDA_FUNC_IN Spectrum sampleDirect(DirectSamplingRecord &dRec, const Vec2f &sample) const
	{
		return sampleDirect_Caller<Spectrum>(*this, dRec, sample);
	}

	CALLER(pdfDirect)
	CUDA_FUNC_IN float pdfDirect(const DirectSamplingRecord &dRec) const
	{
		return pdfDirect_Caller<float>(*this, dRec);
	}

	CALLER(samplePosition)
	CUDA_FUNC_IN Spectrum samplePosition(PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra = 0) const
	{
		return samplePosition_Caller<Spectrum>(*this, pRec, sample, extra);
	}

	CALLER(evalPosition)
	CUDA_FUNC_IN Spectrum evalPosition(const PositionSamplingRecord &pRec) const
	{
		return evalPosition_Caller<Spectrum>(*this, pRec);
	}

	CALLER(pdfPosition)
	CUDA_FUNC_IN float pdfPosition(const PositionSamplingRecord &pRec) const
	{
		return pdfPosition_Caller<float>(*this, pRec);
	}

	CALLER(sampleDirection)
	CUDA_FUNC_IN Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const Vec2f &sample, const Vec2f *extra = 0) const
	{
		return sampleDirection_Caller<Spectrum>(*this, dRec, pRec, sample, extra);
	}

	CALLER(pdfDirection)
	CUDA_FUNC_IN float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		return pdfDirection_Caller<float>(*this, dRec, pRec);
	}

	CALLER(evalDirection)
	CUDA_FUNC_IN Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		return evalDirection_Caller<Spectrum>(*this, dRec, pRec);
	}

	CALLER(getBox)
	CUDA_FUNC_IN AABB getBox(float eps) const
	{
		return getBox_Caller<AABB>(*this, eps);
	}
};

