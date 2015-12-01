#pragma once
#include "Beam.h"
#include <Engine/SpatialGrid.h>

namespace CudaTracerLib {

struct PointStorage : public IVolumeEstimator
{
	struct volPhoton
	{
		Vec3f p;
		Vec3f wi;
		Spectrum phi;
		float rad;
		CUDA_FUNC_IN volPhoton(){}
		CUDA_FUNC_IN volPhoton(const Vec3f& pos, const Vec3f& wi, const Spectrum& phi)
			: p(pos), wi(wi), phi(phi)
		{

		}
	};
	SpatialLinkedMap<volPhoton> m_sStorage;
	unsigned int m_uNumEmitted;
	float m_fCurrentRadiusVol;

	CUDA_FUNC_IN PointStorage()
	{

	}

	PointStorage(unsigned int gridDim, unsigned int numPhotons)
		: m_sStorage(gridDim, numPhotons)
	{

	}

	virtual void Free()
	{
		m_sStorage.Free();
	}

	virtual void StartNewPass(const IRadiusProvider* radProvider, DynamicScene* scene)
	{
		m_fCurrentRadiusVol = radProvider->getCurrentRadius(3);
		m_uNumEmitted = 0;
		m_sStorage.ResetBuffer();
	}

	virtual void StartNewRendering(const AABB& box, float a_InitRadius)
	{
		m_sStorage.SetSceneDimensions(box, a_InitRadius);
	}

	CUDA_FUNC_IN bool isFullK() const
	{
		return m_sStorage.isFull();
	}

	virtual bool isFull() const
	{
		return isFullK();
	}

	virtual unsigned int getNumEmitted() const
	{
		return m_uNumEmitted;
	}

	virtual size_t getSize() const
	{
		return sizeof(*this);
	}

	virtual void PrintStatus(std::vector<std::string>& a_Buf) const
	{
		a_Buf.push_back(format("%.2f%% Vol Photons", (float)m_sStorage.deviceDataIdx / m_sStorage.numData * 100));
	}

	virtual void PrepareForRendering()
	{

	}
#ifdef __CUDACC__
	CUDA_ONLY_FUNC void StoreBeam(const Beam& b, bool firstStore)
	{

	}

	CUDA_ONLY_FUNC void StorePhoton(const Vec3f& pos, const Vec3f& wi, const Spectrum& phi, bool firstStore)
	{
		if (m_sStorage.store(pos, volPhoton(pos, wi, phi)) && firstStore)
			Platform::Increment(&m_uNumEmitted);
	}

	template<bool USE_GLOBAL> CUDA_FUNC_IN Spectrum L_Volume(float a_r, CudaRNG& rng, const Ray& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr)
	{
		Spectrum Tau = Spectrum(0.0f);
		//float Vs = 1.0f / ((4.0f / 3.0f) * PI * a_r * a_r * a_r * m_uNumEmitted), r3 = m_fCurrentRadiusVol * m_fCurrentRadiusVol * m_fCurrentRadiusVol;
		float r3 = m_fCurrentRadiusVol * m_fCurrentRadiusVol * m_fCurrentRadiusVol, Vs = 1.0f / (m_uNumEmitted * r3 * 4.0f / 3.0f * PI);
		Spectrum L_n = Spectrum(0.0f);
		float a, b;
		if (!m_sStorage.hashMap.getAABB().Intersect(r, &a, &b))
			return L_n;//that would be dumb
		float minT = a = math::clamp(a, tmin, tmax);
		b = math::clamp(b, tmin, tmax);
		float d = 2.0f * a_r;
		while (a < b)
		{
			float t = a + d / 2.0f;
			Vec3f x = r(t);
			m_sStorage.ForAll(x - Vec3f(a_r), x + Vec3f(a_r), [&](unsigned int p_idx, const volPhoton& ph)
			{
				if (distanceSquared(ph.p, x) < r3)
				{
					float p = vol.p(x, r.direction, ph.wi, rng);
					//float l1 = dot(ph.p - r.origin, r.direction) / dot(r.direction, r.direction);
					//Spectrum tauToPhoton = (-Tau - vol.tau(r, a, l1)).exp();
					//L_n += p * ph.phi * Vs * tauToPhoton * d;
					Spectrum camera_sc = vol.sigma_s(x, r.direction);
					L_n += p * ph.phi * Vs * camera_sc;
				}
			});
			Spectrum tauDelta = vol.tau(r, a, a + d);
			Tau += tauDelta;
			L_n += vol.Lve(x, -1.0f * r.direction) * d;
			a += d;
		}
		Tr = (-Tau).exp();
		return L_n;
	}
#endif
};

}
