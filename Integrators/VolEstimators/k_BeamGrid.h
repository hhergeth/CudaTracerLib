#pragma once
#include "k_PointStorage.h"

namespace CudaTracerLib {

struct k_BeamGrid : public k_PointStorage
{
	e_SpatialLinkedMap<int> m_sBeamGridStorage;

	int photonDensNum;

	CUDA_FUNC_IN k_BeamGrid()
	{

	}

	k_BeamGrid(unsigned int gridDim, unsigned int numPhotons, int N = 20, float nnSearch = 1)
		: k_PointStorage(gridDim, numPhotons), photonDensNum(nnSearch), m_sBeamGridStorage(gridDim, gridDim * gridDim * gridDim * (1 + N))
	{

	}

	virtual void Free()
	{
		k_PointStorage::Free();
		m_sBeamGridStorage.Free();
	}

	virtual void StartNewPass(const IRadiusProvider* radProvider, e_DynamicScene* scene)
	{
		k_PointStorage::StartNewPass(radProvider, scene);
		m_sBeamGridStorage.ResetBuffer();
	}

	virtual void StartNewRendering(const AABB& box, float a_InitRadius)
	{
		k_PointStorage::StartNewRendering(box, a_InitRadius);
		m_sBeamGridStorage.SetSceneDimensions(box, a_InitRadius);
	}

	virtual size_t getSize() const
	{
		return sizeof(*this);
	}

	virtual void PrepareForRendering();

	virtual void PrintStatus(std::vector<std::string>& a_Buf) const
	{
		k_PointStorage::PrintStatus(a_Buf);
		a_Buf.push_back(format("%.2f%% Beam indices", (float)m_sBeamGridStorage.deviceDataIdx / m_sBeamGridStorage.numData * 100));
	}

	template<bool USE_GLOBAL> CUDA_FUNC_IN Spectrum L_Volume(float a_r, CudaRNG& rng, const Ray& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr)
	{
		Spectrum Tau = Spectrum(0.0f);
		float r2 = a_r * a_r;
		Spectrum L_n = Spectrum(0.0f);
		TraverseGrid(r, m_sStorage.hashMap, tmin, tmax, [&](float minT, float rayT, float maxT, float cellEndT, Vec3u& cell_pos, bool& cancelTraversal)
		{
			m_sBeamGridStorage.ForAll(cell_pos, [&](unsigned int, unsigned int beam_idx)
			{
				const volPhoton& ph = m_sStorage(beam_idx);
				float l1 = dot(ph.p - r.origin, r.direction) / dot(r.direction, r.direction);
				if (distanceSquared(ph.p, r(l1)) < ph.rad && rayT <= l1 && l1 <= cellEndT)
				{
					float p = vol.p(ph.p, r.direction, ph.wi, rng);
					Spectrum tauToPhoton = (-Tau - vol.tau(r, rayT, l1)).exp();
					L_n += p * ph.phi / (PI * m_uNumEmitted * ph.rad) * tauToPhoton;
				}
			});
			float localDist = cellEndT - rayT;
			Spectrum tauD = vol.tau(r, rayT, cellEndT);
			Tau += tauD;
			L_n += vol.Lve(r(rayT + localDist / 2), -1.0f * r.direction) * localDist;
		});
		Tr = (-Tau).exp();
		return L_n;
	}
};

}