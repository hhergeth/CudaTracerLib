#pragma once
#include <Defines.h>
#include "../AdaptiveHelper.h"
#include <Kernel/TraceHelper.h>
#include <vector>
#include <Base/Platform.h>
#include <Engine/Grid.h>
#include <Math/Compression.h>
#include <Engine/SynchronizedBuffer.h>

namespace CudaTracerLib {

#ifndef __CUDACC__
inline unsigned int atomicInc(unsigned int* i, unsigned int j)
{
	return Platform::Increment(i);
}
#endif

class IRadiusProvider
{
public:
	virtual float getCurrentRadius(float exp) const = 0;
};

class IVolumeEstimator : public ISynchronizedBufferParent
{
public:
	template<typename... ARGS> IVolumeEstimator(ARGS&... args)
		: ISynchronizedBufferParent(args...)
	{
	}

	virtual void StartNewPass(const IRadiusProvider* radProvider, DynamicScene* scene) = 0;

	virtual void StartNewRendering(const AABB& box) = 0;

	virtual bool isFull() const = 0;

	virtual void PrepareForRendering() = 0;

	virtual size_t getSize() const = 0;

	virtual void PrintStatus(std::vector<std::string>& a_Buf) const = 0;

	virtual void getStatusInfo(size_t& length, size_t& count) const = 0;
};

template<bool USE_GLOBAL> struct VolHelper
{
	const VolumeRegion* vol;

	CUDA_FUNC_IN VolHelper(const VolumeRegion* v = 0)
		: vol(v)
	{

	}

	CUDA_FUNC_IN bool IntersectP(const Ray &ray, float minT, float maxT, float *t0, float *t1) const
	{
		if (USE_GLOBAL)
			return g_SceneData.m_sVolume.IntersectP(ray, minT, maxT, t0, t1);
		else return vol->IntersectP(ray, minT, maxT, t0, t1);
	}

	CUDA_FUNC_IN Spectrum sigma_s(const Vec3f& p, const NormalizedT<Vec3f>& w) const
	{
		if (USE_GLOBAL)
			return g_SceneData.m_sVolume.sigma_s(p, w);
		else return vol->sigma_s(p, w);
	}

	CUDA_FUNC_IN Spectrum sigma_a(const Vec3f& p, const NormalizedT<Vec3f>& w) const
	{
		if (USE_GLOBAL)
			return g_SceneData.m_sVolume.sigma_a(p, w);
		else return vol->sigma_a(p, w);
	}

	CUDA_FUNC_IN Spectrum sigma_t(const Vec3f& p, const NormalizedT<Vec3f>& w) const
	{
		if (USE_GLOBAL)
			return g_SceneData.m_sVolume.sigma_t(p, w);
		else return vol->sigma_t(p, w);
	}

	CUDA_FUNC_IN Spectrum Lve(const Vec3f& p, const NormalizedT<Vec3f>& w) const
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

	CUDA_FUNC_IN float p(const Vec3f& p, const PhaseFunctionSamplingRecord& pRec) const
	{
		if (USE_GLOBAL)
			return g_SceneData.m_sVolume.p(p, pRec);
		else return vol->As()->Func.Evaluate(pRec);
	}
};

}
