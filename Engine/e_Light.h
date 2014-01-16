#pragma once

#include "..\Base\CudaRandom.h"
#include <MathTypes.h>
#include "e_ShapeSet.h"
#include "e_KernelDynamicScene.h"

struct e_LightBase : public e_BaseType
{
	bool IsDelta;

	e_LightBase(bool d)
		: IsDelta(d)
	{
	}
};

#define e_PointLight_TYPE 1
struct e_PointLight : public e_LightBase
{
	float3 lightPos;
    Spectrum m_intensity;
	
	e_PointLight()
		: e_LightBase(true)
	{}
	e_PointLight(float3 p, Spectrum L, float r = 0)
		: e_LightBase(true), lightPos(p), m_intensity(L)
	{

	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleRay(Ray &ray, const float2 &spatialSample, const float2 &directionalSample) const;

	CUDA_FUNC_IN Spectrum eval(const float3& p, const Frame& sys, const float3 &d) const
	{
		return Spectrum(0.0f);
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirect(DirectSamplingRecord &dRec, const float2 &sample) const;

	CUDA_FUNC_IN float pdfDirect(const DirectSamplingRecord &dRec) const
	{
		return dRec.measure == EDiscrete ? 1.0f : 0.0f;
	}

	CUDA_DEVICE CUDA_HOST Spectrum samplePosition(PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const;

	CUDA_FUNC_IN Spectrum evalPosition(const PositionSamplingRecord &pRec) const
	{
		return (pRec.measure == EDiscrete) ? (m_intensity * 4*PI) : Spectrum(0.0f);
	}

	CUDA_FUNC_IN float pdfPosition(const PositionSamplingRecord &pRec) const
	{
		return (pRec.measure == EDiscrete) ? 1.0f : 0.0f;
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const;

	CUDA_FUNC_IN float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		return (dRec.measure == ESolidAngle) ? INV_FOURPI : 0.0f;
	}

	CUDA_FUNC_IN Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		return Spectrum((dRec.measure == ESolidAngle) ? INV_FOURPI : 0.0f);
	}

	AABB getBox(float eps) const
	{
		return AABB(lightPos - make_float3(eps), lightPos + make_float3(eps));
	}
	
	TYPE_FUNC(e_PointLight)
};

#define e_DiffuseLight_TYPE 2
struct e_DiffuseLight : public e_LightBase
{
	Spectrum m_radiance, m_power;
    ShapeSet shapeSet;
	
	e_DiffuseLight()
		: e_LightBase(false)
	{}
	e_DiffuseLight(const Spectrum& L, ShapeSet& s)
		: e_LightBase(false), shapeSet(s)
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

	void Recalculate(const float4x4& mat)
	{
		shapeSet.Recalculate(mat);
		Update();
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleRay(Ray &ray, const float2 &spatialSample, const float2 &directionalSample) const;

	CUDA_DEVICE CUDA_HOST Spectrum eval(const float3& p, const Frame& sys, const float3 &d) const;

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirect(DirectSamplingRecord &dRec, const float2 &sample) const;

	CUDA_DEVICE CUDA_HOST float pdfDirect(const DirectSamplingRecord &dRec) const;

	CUDA_FUNC_IN Spectrum samplePosition(PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const
	{
		shapeSet.SamplePosition(pRec, sample);
		return m_power;
	}

	CUDA_FUNC_IN Spectrum evalPosition(const PositionSamplingRecord &pRec) const
	{
		return m_radiance * PI;
	}

	CUDA_FUNC_IN float pdfPosition(const PositionSamplingRecord &pRec) const
	{
		return shapeSet.Pdf(pRec);
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const;

	CUDA_DEVICE CUDA_HOST float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const;

	CUDA_DEVICE CUDA_HOST Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const;

	AABB getBox(float eps) const
	{
		return shapeSet.getBox();
	}
	
	TYPE_FUNC(e_DiffuseLight)
};

#define e_DistantLight_TYPE 3
struct e_DistantLight : public e_LightBase
{
	Spectrum m_normalIrradiance, m_power;
	Frame ToWorld;
	float m_invSurfaceArea, radius;
	
	e_DistantLight()
		: e_LightBase(true)
	{}
	///r is the radius of the scene's bounding sphere
	e_DistantLight(const Spectrum& L, float3 d, float r)
		: e_LightBase(true), ToWorld(d), radius(r * 1.1f)
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

	CUDA_DEVICE CUDA_HOST Spectrum sampleRay(Ray &ray, const float2 &spatialSample, const float2 &directionalSample) const;

	CUDA_FUNC_IN Spectrum eval(const float3& p, const Frame& sys, const float3 &d) const
	{
		return Spectrum(0.0f);
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirect(DirectSamplingRecord &dRec, const float2 &sample) const;

	CUDA_FUNC_IN float pdfDirect(const DirectSamplingRecord &dRec) const
	{
		return dRec.measure == EDiscrete ? 1.0f : 0.0f;
	}

	CUDA_DEVICE CUDA_HOST Spectrum samplePosition(PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const;

	CUDA_FUNC_IN Spectrum evalPosition(const PositionSamplingRecord &pRec) const
	{
		return (pRec.measure == EArea) ? m_normalIrradiance : Spectrum(0.0f);
	}

	CUDA_FUNC_IN float pdfPosition(const PositionSamplingRecord &pRec) const
	{
		return (pRec.measure == EArea) ? m_invSurfaceArea : 0.0f;
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const;

	CUDA_FUNC_IN float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		return (dRec.measure == EDiscrete) ? 1.0f : 0.0f;
	}

	CUDA_FUNC_IN Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		return Spectrum((dRec.measure == EDiscrete) ? 1.0f : 0.0f);
	}
	
	AABB getBox(float eps) const
	{
		return AABB(make_float3(-radius), make_float3(+radius));
	}
	
	TYPE_FUNC(e_DistantLight)
};

#define e_SpotLight_TYPE 4
struct e_SpotLight : public e_LightBase
{
    Spectrum m_intensity;
	float m_beamWidth, m_cutoffAngle;
	float m_cosBeamWidth, m_cosCutoffAngle, m_invTransitionWidth;
	Frame ToWorld;
	float3 Position, Target;
	
	e_SpotLight()
		: e_LightBase(true)
	{}
	e_SpotLight(float3 p, float3 t, Spectrum L, float width, float fall);

	virtual void Update()
	{
		ToWorld = Frame(Target - Position);
		m_cosBeamWidth = cosf(m_beamWidth);
		m_cosCutoffAngle = cosf(m_cutoffAngle);
		m_invTransitionWidth = 1.0f / (m_cutoffAngle - m_beamWidth);
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleRay(Ray &ray, const float2 &spatialSample, const float2 &directionalSample) const;

	CUDA_FUNC_IN Spectrum eval(const float3& p, const Frame& sys, const float3 &d) const
	{
		return Spectrum(0.0f);
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirect(DirectSamplingRecord &dRec, const float2 &sample) const;

	CUDA_FUNC_IN float pdfDirect(const DirectSamplingRecord &dRec) const
	{
		return dRec.measure == EDiscrete ? 1.0f : 0.0f;
	}

	CUDA_DEVICE CUDA_HOST Spectrum samplePosition(PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const;

	CUDA_FUNC_IN Spectrum evalPosition(const PositionSamplingRecord &pRec) const
	{
		return (pRec.measure == EDiscrete) ? (m_intensity * 4*PI) : Spectrum(0.0f);
	}

	CUDA_FUNC_IN float pdfPosition(const PositionSamplingRecord &pRec) const
	{
		return (pRec.measure == EDiscrete) ? 1.0f : 0.0f;
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const;

	CUDA_FUNC_IN float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		return (dRec.measure == ESolidAngle) ? Warp::squareToUniformConePdf(m_cosCutoffAngle) : 0.0f;
	}

	CUDA_FUNC_IN Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		return (dRec.measure == ESolidAngle) ? falloffCurve(ToWorld.toLocal(dRec.d)) * INV_FOURPI : Spectrum(0.0f);
	}
	
	AABB getBox(float eps) const
	{
		return AABB(Position - make_float3(eps), Position + make_float3(eps));
	}
	
	TYPE_FUNC(e_SpotLight)
private:
	CUDA_DEVICE CUDA_HOST Spectrum falloffCurve(const float3 &d) const;
};

#define e_InfiniteLight_TYPE 5
struct e_InfiniteLight : public e_LightBase
{
	e_KernelMIPMap radianceMap;
	e_Variable<float> m_cdfRows, m_cdfCols, m_rowWeights;
	float3 m_SceneCenter;
	float m_SceneRadius;
	float m_normalization;
	float m_power;
	float m_invSurfaceArea;
	float2 m_size, m_pixelSize;
	Spectrum m_scale;
	
	e_InfiniteLight()
		: e_LightBase(false)
	{}
	CUDA_HOST e_InfiniteLight(e_Stream<char>* a_Buffer, e_BufferReference<e_MIPMap, e_KernelMIPMap>& mip, const Spectrum& scale, const AABB& scenBox);

	CUDA_DEVICE CUDA_HOST Spectrum sampleRay(Ray &ray, const float2 &spatialSample, const float2 &directionalSample) const;

	CUDA_FUNC_IN Spectrum eval(const float3& p, const Frame& sys, const float3 &d) const
	{
		return evalEnvironment(Ray(p, -1.0f * d));
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirect(DirectSamplingRecord &dRec, const float2 &sample) const;

	CUDA_DEVICE CUDA_HOST float pdfDirect(const DirectSamplingRecord &dRec) const;

	CUDA_DEVICE CUDA_HOST Spectrum samplePosition(PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const;

	CUDA_FUNC_IN Spectrum evalPosition(const PositionSamplingRecord &pRec) const
	{
		return Spectrum(m_power * m_invSurfaceArea);
	}

	CUDA_FUNC_IN float pdfPosition(const PositionSamplingRecord &pRec) const
	{
		return m_invSurfaceArea;
	}

	CUDA_DEVICE CUDA_HOST Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra) const;

	CUDA_FUNC_IN float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		return internalPdfDirection(-1.0f * dRec.d);
	}

	CUDA_DEVICE CUDA_HOST Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const;

	CUDA_DEVICE CUDA_HOST Spectrum evalEnvironment(const Ray &ray) const;
	
	AABB getBox(float eps) const
	{
		return AABB(-make_float3(1.0f / eps), make_float3(1.0f / eps));
	}

	TYPE_FUNC(e_InfiniteLight)
private:
	CUDA_DEVICE CUDA_HOST void internalSampleDirection(float2 sample, float3 &d, Spectrum &value, float &pdf) const;
	CUDA_DEVICE CUDA_HOST float internalPdfDirection(const float3 &d) const;
	CUDA_DEVICE CUDA_HOST unsigned int sampleReuse(float *cdf, unsigned int size, float &sample) const;
};

#define LGT_SIZE RND_16(DMAX5(sizeof(e_PointLight), sizeof(e_DiffuseLight), sizeof(e_DistantLight), sizeof(e_SpotLight), sizeof(e_InfiniteLight)))

CUDA_ALIGN(16) struct e_KernelLight : public e_AggregateBaseType<e_LightBase, LGT_SIZE>
{
public:
	CUDA_FUNC_IN Spectrum sampleRay(Ray &ray, const float2 &spatialSample, const float2 &directionalSample) const
	{
		CALL_FUNC5(e_PointLight,e_DiffuseLight,e_DistantLight,e_SpotLight,e_InfiniteLight, sampleRay(ray, spatialSample, directionalSample))
		return 0.0f;
	}

	CUDA_FUNC_IN Spectrum eval(const float3& p, const Frame& sys, const float3 &d) const
	{
		CALL_FUNC5(e_PointLight,e_DiffuseLight,e_DistantLight,e_SpotLight,e_InfiniteLight, eval(p, sys, d))
		return 0.0f;
	}

	CUDA_FUNC_IN Spectrum sampleDirect(DirectSamplingRecord &dRec, const float2 &sample) const
	{
		CALL_FUNC5(e_PointLight,e_DiffuseLight,e_DistantLight,e_SpotLight,e_InfiniteLight, sampleDirect(dRec, sample))
		return 0.0f;
	}

	CUDA_FUNC_IN float pdfDirect(const DirectSamplingRecord &dRec) const
	{
		CALL_FUNC5(e_PointLight,e_DiffuseLight,e_DistantLight,e_SpotLight,e_InfiniteLight, pdfDirect(dRec))
		return 0.0f;
	}

	CUDA_FUNC_IN Spectrum samplePosition(PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra = 0) const
	{
		CALL_FUNC5(e_PointLight,e_DiffuseLight,e_DistantLight,e_SpotLight,e_InfiniteLight, samplePosition(pRec, sample, extra))
		return 0.0f;
	}

	CUDA_FUNC_IN Spectrum evalPosition(const PositionSamplingRecord &pRec) const
	{
		CALL_FUNC5(e_PointLight,e_DiffuseLight,e_DistantLight,e_SpotLight,e_InfiniteLight, evalPosition(pRec))
		return 0.0f;
	}

	CUDA_FUNC_IN float pdfPosition(const PositionSamplingRecord &pRec) const
	{
		CALL_FUNC5(e_PointLight,e_DiffuseLight,e_DistantLight,e_SpotLight,e_InfiniteLight, pdfPosition(pRec))
		return 0.0f;
	}

	CUDA_FUNC_IN Spectrum sampleDirection(DirectionSamplingRecord &dRec, PositionSamplingRecord &pRec, const float2 &sample, const float2 *extra = 0) const
	{
		CALL_FUNC5(e_PointLight,e_DiffuseLight,e_DistantLight,e_SpotLight,e_InfiniteLight, sampleDirection(dRec, pRec, sample, extra))
		return 0.0f;
	}

	CUDA_FUNC_IN float pdfDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		CALL_FUNC5(e_PointLight,e_DiffuseLight,e_DistantLight,e_SpotLight,e_InfiniteLight, pdfDirection(dRec, pRec))
		return 0.0f;
	}

	CUDA_FUNC_IN Spectrum evalDirection(const DirectionSamplingRecord &dRec, const PositionSamplingRecord &pRec) const
	{
		CALL_FUNC5(e_PointLight,e_DiffuseLight,e_DistantLight,e_SpotLight,e_InfiniteLight, evalDirection(dRec, pRec))
		return 0.0f;
	}

	CUDA_FUNC_IN bool IsDeltaLight() const
	{
		return ((e_LightBase*)Data)->IsDelta;
	}

	AABB getBox(float eps) const
	{
		CALL_FUNC5(e_PointLight,e_DiffuseLight,e_DistantLight,e_SpotLight,e_InfiniteLight, getBox(eps))
		return AABB::Identity();
	}
};

