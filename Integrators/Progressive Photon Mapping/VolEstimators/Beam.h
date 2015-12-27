#pragma once
#include <Defines.h>
#include <MathTypes.h>
#include <Kernel/TraceHelper.h>
#include <vector>
#include <Base/Platform.h>
#include <Engine/Grid.h>
#include <Math/Compression.h>

namespace CudaTracerLib {

struct Beam
{
	unsigned short dir;
	RGBE phi;
	float t;
	Vec3f pos;

	CUDA_FUNC_IN Beam(){}
	CUDA_FUNC_IN Beam(const Vec3f& p, const Vec3f& d, float t, const Spectrum& ph)
		: pos(p), dir(NormalizedFloat3ToUchar2(d)), t(t), phi(ph.toRGBE())
	{

	}

	CUDA_FUNC_IN Vec3f getPos() const
	{
		return pos;
	}
	CUDA_FUNC_IN Vec3f getDir() const
	{
		return Uchar2ToNormalizedFloat3(dir);
	}
	CUDA_FUNC_IN Spectrum getL() const
	{
		Spectrum s;
		s.fromRGBE(phi);
		return s;
	}

	CUDA_FUNC_IN AABB getAABB(float r) const
	{
		const Vec3f beamStart = getPos();
		const Vec3f beamEnd = beamStart + dir * t;
		const Vec3f startMargins(r);
		const Vec3f endMargins(r);
		const Vec3f minPt = min(beamStart - startMargins, beamEnd - endMargins);
		const Vec3f maxPt = max(beamStart + startMargins, beamEnd + endMargins);
		return AABB(minPt, maxPt);
	}

	CUDA_FUNC_IN AABB getSegmentAABB(float splitMin, float splitMax, float r) const
	{
		splitMin *= t;
		splitMax *= t;
		const Vec3f P = getPos();
		const Vec3f beamStart = P + dir * splitMin;
		const Vec3f beamEnd = P + dir * splitMax;
		const Vec3f startMargins(r);
		const Vec3f endMargins(r);
		const Vec3f minPt = min(beamStart - startMargins, beamEnd - endMargins);
		const Vec3f maxPt = max(beamStart + startMargins, beamEnd + endMargins);
		return AABB(minPt, maxPt);
	}

	CUDA_FUNC_IN static bool testIntersectionBeamBeam(
		const Vec3f& O1,
		const Vec3f& d1,
		const float minT1,
		const float maxT1,
		const Vec3f& O2,
		const Vec3f& d2,
		const float minT2,
		const float maxT2,
		const float maxDistSqr,
		float& oDistance,
		float& oSinTheta,
		float& oT1,
		float& oT2)
	{
		const Vec3f  d1d2c = cross(d1, d2);
		const float sinThetaSqr = dot(d1d2c, d1d2c); // Square of the sine between the two lines (||cross(d1, d2)|| = sinTheta).

		const float ad = dot((O2 - O1), d1d2c);

		// Lines too far apart.
		if (ad*ad >= maxDistSqr*sinThetaSqr)//multiply 1/l * 1/l to the rhs, l = sqrt(sinThetaSqr)
			return false;

		// Cosine between the two lines.
		const float d1d2 = dot(d1, d2);
		const float d1d2Sqr = d1d2*d1d2;
		const float d1d2SqrMinus1 = d1d2Sqr - 1.0f;

		// Parallel lines?
		if (d1d2SqrMinus1 < 1e-5f && d1d2SqrMinus1 > -1e-5f)
			return false;

		const float d1O1 = dot(d1, O1);
		const float d1O2 = dot(d1, O2);

		oT1 = (d1O1 - d1O2 - d1d2 * (dot(d2, O1) - dot(d2, O2))) / d1d2SqrMinus1;

		// Out of range on ray 1.
		if (oT1 <= minT1 || oT1 >= maxT1)
			return false;

		oT2 = (oT1 + d1O1 - d1O2) / d1d2;
		// Out of range on ray 2.
		if (oT2 <= minT2 || oT2 >= maxT2 || isnan(oT2))
			return false;

		const float sinTheta = math::sqrt(sinThetaSqr);

		oDistance = math::abs(ad) / sinTheta;

		oSinTheta = sinTheta;

		return true; // Found an intersection.
	}
};

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

class IVolumeEstimator
{
public:
	virtual void Free()
	{

	}

	virtual void StartNewPass(const IRadiusProvider* radProvider, DynamicScene* scene) = 0;

	virtual void StartNewRendering(const AABB& box) = 0;

	virtual bool isFull() const = 0;

	virtual void PrepareForRendering() = 0;

	virtual size_t getSize() const = 0;

	virtual void PrintStatus(std::vector<std::string>& a_Buf) const = 0;

	virtual unsigned int getNumEmitted() const = 0;
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
