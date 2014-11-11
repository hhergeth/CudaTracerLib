#pragma once

#include "..\Math\AABB.h"
#include "e_Buffer.h"
#include "e_PhaseFunction.h"

struct e_BaseVolumeRegion : public e_BaseType
{
public:
	AABB Box;
	e_PhaseFunction Func;
	CUDA_FUNC_IN bool inside(const float3& p) const
	{
		return Box.Contains(p);
	}
};

#define e_HomogeneousVolumeDensity_TYPE 1
struct e_HomogeneousVolumeDensity : public e_BaseVolumeRegion
{
public:
	e_HomogeneousVolumeDensity(){}
	e_HomogeneousVolumeDensity(const float sa, const float ss, const e_PhaseFunction& func, float emit, const AABB& box)
	{
		e_BaseVolumeRegion::Box = box;
		e_BaseVolumeRegion::Func = func;
        WorldToVolume = float4x4::Identity();
        sig_a = Spectrum(sa);
        sig_s = Spectrum(ss);
        le = Spectrum(emit);
	}

	e_HomogeneousVolumeDensity(const Spectrum& sa, const Spectrum& ss, const e_PhaseFunction& func, const Spectrum& emit, const AABB& box, const float4x4& v2w)
	{
		e_BaseVolumeRegion::Box = box;
		e_BaseVolumeRegion::Func = func;
        WorldToVolume = v2w.inverse();
        sig_a = sa;
        sig_s = ss;
        le = emit;
	}

    CUDA_DEVICE CUDA_HOST bool IntersectP(const Ray &ray, const float minT, const float maxT, float *t0, float *t1) const;

    CUDA_FUNC_IN Spectrum sigma_a(const float3& p, const float3& w) const
	{
		return inside(p) ? sig_a : Spectrum(0.0f);
	}

    CUDA_FUNC_IN Spectrum sigma_s(const float3& p, const float3& w) const
	{
		return inside(p) ? sig_s : Spectrum(0.0f);
	}

    CUDA_FUNC_IN Spectrum Lve(const float3& p, const float3& w) const
	{
		return inside(p) ? le : Spectrum(0.0f);
	}

    CUDA_FUNC_IN Spectrum sigma_t(const float3 &p, const float3 &wo) const
	{
		return inside(p) ? (sig_s + sig_a) : Spectrum(0.0f);
	}

    CUDA_DEVICE CUDA_HOST Spectrum tau(const Ray &ray, const float minT, const float maxT, float step = 1.f, float offset = 0.5) const;

	TYPE_FUNC(e_HomogeneousVolumeDensity)
public:
	Spectrum sig_a, sig_s, le;
	float4x4 WorldToVolume;
};

#define VOL_SIZE RND_16((sizeof(e_HomogeneousVolumeDensity)))

struct CUDA_ALIGN(16) e_VolumeRegion : public e_AggregateBaseType<e_BaseVolumeRegion, VOL_SIZE> 
{
public:
	CUDA_FUNC_IN e_VolumeRegion()
	{
		type = 0;
	}

	CUDA_FUNC_IN e_BaseVolumeRegion* BaseRegion()
	{
		return (e_BaseVolumeRegion*)Data;
	}

	CUDA_FUNC_IN AABB WorldBound()
	{
		return ((e_BaseVolumeRegion*)Data)->Box;
	}

	CUDA_FUNC_IN bool IntersectP(const Ray &ray, const float minT, const float maxT, float *t0, float *t1) const
	{
		CALL_FUNC1(e_HomogeneousVolumeDensity, IntersectP(ray, minT, maxT, t0, t1))
		return false;
	}

    CUDA_FUNC_IN Spectrum sigma_a(const float3& p, const float3& w) const
	{
		CALL_FUNC1(e_HomogeneousVolumeDensity, sigma_a(p, w))
		return 0.0f;
	}

    CUDA_FUNC_IN Spectrum sigma_s(const float3& p, const float3& w) const
	{
		CALL_FUNC1(e_HomogeneousVolumeDensity, sigma_s(p, w))
		return 0.0f;
	}

    CUDA_FUNC_IN Spectrum Lve(const float3& p, const float3& w) const
	{
		CALL_FUNC1(e_HomogeneousVolumeDensity, Lve(p, w))
		return 0.0f;
	}

    CUDA_FUNC_IN Spectrum sigma_t(const float3 &p, const float3 &wo) const
	{
		CALL_FUNC1(e_HomogeneousVolumeDensity, sigma_t(p, wo))
		return 0.0f;
	}

    CUDA_FUNC_IN Spectrum tau(const Ray &ray, float minT, float maxT, float step = 1.f, float offset = 0.5) const
	{
		CALL_FUNC1(e_HomogeneousVolumeDensity, tau(ray, minT, maxT, step, offset))
		return 0.0f;
	}
};

struct e_KernelAggregateVolume
{
public:
	unsigned int m_uVolumeCount;
	e_VolumeRegion* m_pVolumes;
	AABB box;
public:
	CUDA_FUNC_IN e_KernelAggregateVolume()
	{

	}
	e_KernelAggregateVolume(e_Stream<e_VolumeRegion>* D, bool devicePointer = true)
	{
		m_uVolumeCount = D->UsedElements().getLength();
		m_pVolumes = D->getKernelData(devicePointer).Data;
		box = AABB::Identity();
		for(unsigned int i = 0; i < m_uVolumeCount; i++)
			box.Enlarge(D->operator()(i)->WorldBound());
	}

	///Calculates the intersection of the ray with the bound of the volume
    CUDA_DEVICE CUDA_HOST bool IntersectP(const Ray &ray, const float minT, const float maxT, float *t0, float *t1) const;

	///The probability that light is abosrbed per unit distance
    CUDA_DEVICE CUDA_HOST Spectrum sigma_a(const float3& p, const float3& w) const;

	///The probability that light is scattered per unit distance
    CUDA_DEVICE CUDA_HOST Spectrum sigma_s(const float3& p, const float3& w) const;

    CUDA_DEVICE CUDA_HOST Spectrum Lve(const float3& p, const float3& w) const;

	///Combined sigmas
    CUDA_DEVICE CUDA_HOST Spectrum sigma_t(const float3 &p, const float3 &wo) const;

	///Calculates the volumes optical thickness along a ray in the volumes bounds
    CUDA_DEVICE CUDA_HOST Spectrum tau(const Ray &ray, const float minT, const float maxT, float step = 1.f, float offset = 0.5) const;

	CUDA_DEVICE CUDA_HOST float Sample(const float3& p, const float3& wo, CudaRNG& rng, float3* wi);

	CUDA_DEVICE CUDA_HOST float p(const float3& p, const float3& wo, const float3& wi, CudaRNG& rng);

	CUDA_FUNC_IN bool HasVolumes()
	{
		return m_uVolumeCount > 0;
	}
};