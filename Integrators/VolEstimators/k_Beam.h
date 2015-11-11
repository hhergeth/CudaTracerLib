#pragma once
#include <Defines.h>
#include <MathTypes.h>
#include <Kernel/k_TraceHelper.h>
#include <vector>
#include <Base/Platform.h>

namespace CudaTracerLib {

struct k_Beam
{
	Vec3f pos;
	Vec3f dir;
	float t;
	Spectrum Phi;
	unsigned int lastEntry;
	k_Beam(){}
	CUDA_FUNC_IN k_Beam(const Vec3f& p, const Vec3f& d, float t, const Spectrum& ph)
		: pos(p), dir(d), t(t), Phi(ph), lastEntry(0)
	{

	}
};

#ifndef __CUDACC__
inline unsigned int atomicInc(unsigned int* i, unsigned int j)
{
	return Platform::Increment(i);
}
#endif

CUDA_FUNC_IN float skew_lines(const Ray& r, const Ray& r2, float& t1, float& t2)
{
	if (absdot(r.direction.normalized(), r2.direction.normalized()) > 1 - 1e-2f)
		return FLT_MAX;

	float v1dotv2 = dot(r.direction, r2.direction), v1p2 = r.direction.lenSqr(), v2p2 = r2.direction.lenSqr();
	float x = dot(r2.origin - r.origin, r.direction), y = dot(r2.origin - r.origin, r2.direction);
	float dc = 1.0f / (v1dotv2 * v1dotv2 - v1p2 * v2p2);
	t1 = dc * (-v2p2 * x + v1dotv2 * y);
	t2 = dc * (-v1dotv2 * x + v1p2 * y);

	float D = math::abs(dot(cross(r.direction, r2.direction).normalized(), r.origin - r2.origin));
	float d = (r(t1) - r2(t2)).length();
	float err = math::abs(D - d);
	if (err > 0.1f)
		printf("D = %f, d = %f, r1 = {(%f,%f,%f), (%f,%f,%f)}, r2 = {(%f,%f,%f), (%f,%f,%f)}\n", D, d, r.origin.x, r.origin.y, r.origin.z, r.direction.x, r.direction.y, r.direction.z, r2.origin.x, r2.origin.y, r2.origin.z, r2.direction.x, r2.direction.y, r2.direction.z);

	return D;
}

class IRadiusProvider
{
public:
	virtual float getCurrentRadius(float exp) const = 0;
};

class IVolumeEstimator
{
public:
	virtual void Free()
	{

	}

	virtual void StartNewPass(const IRadiusProvider* radProvider, e_DynamicScene* scene) = 0;

	virtual void StartNewRendering(const AABB& box, float a_InitRadius) = 0;

	virtual bool isFull() const = 0;

	virtual void PrepareForRendering() = 0;

	virtual size_t getSize() const = 0;

	virtual void PrintStatus(std::vector<std::string>& a_Buf) const = 0;

	virtual unsigned int getNumEmitted() const = 0;
};

template<bool USE_GLOBAL> struct VolHelper
{
	const e_VolumeRegion* vol;

	CUDA_FUNC_IN VolHelper(const e_VolumeRegion* v = 0)
		: vol(v)
	{

	}

	CUDA_FUNC_IN bool IntersectP(const Ray &ray, float minT, float maxT, float *t0, float *t1) const
	{
		if (USE_GLOBAL)
			return g_SceneData.m_sVolume.IntersectP(ray, minT, maxT, t0, t1);
		else return vol->IntersectP(ray, minT, maxT, t0, t1);
	}

	CUDA_FUNC_IN Spectrum sigma_s(const Vec3f& p, const Vec3f& w) const
	{
		if (USE_GLOBAL)
			return g_SceneData.m_sVolume.sigma_s(p, w);
		else return vol->sigma_s(p, w);
	}

	CUDA_FUNC_IN Spectrum Lve(const Vec3f& p, const Vec3f& w) const
	{
		if (USE_GLOBAL)
			return g_SceneData.m_sVolume.Lve(p, w);
		else return vol->Lve(p, w);
	}

	CUDA_FUNC_IN Spectrum tau(const Ray &ray, float minT, float maxT) const
	{
		if (USE_GLOBAL)
			return g_SceneData.m_sVolume.tau(ray, minT, maxT);
		else return vol->tau(ray, minT, maxT);
	}

	CUDA_FUNC_IN float p(const Vec3f& p, const Vec3f& wo, const Vec3f& wi, CudaRNG& rng) const
	{
		PhaseFunctionSamplingRecord rec(wo, wi);
		if (USE_GLOBAL)
			return g_SceneData.m_sVolume.p(p, wo, wi, rng);
		else return vol->As()->Func.Evaluate(rec);
	}
};

}