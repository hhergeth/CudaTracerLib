#pragma once
#include "k_PointStorage.h"

struct k_BeamGrid : public k_PointStorage
{
	unsigned int m_uIndex;
	unsigned int m_uNumEntries, m_uGridEntries;
	Vec2i* m_pDeviceData;

	int photonDensNum;

	CUDA_FUNC_IN k_BeamGrid()
	{

	}

	k_BeamGrid(unsigned int gridDim, unsigned int numPhotons)
		: k_PointStorage(gridDim, numPhotons)
	{
		const int N = 20;
		photonDensNum = 2;

		m_uGridEntries = gridDim*gridDim*gridDim;
		m_uNumEntries = m_uGridEntries * (1 + N);
		CUDA_MALLOC(&m_pDeviceData, sizeof(Vec2i) * m_uNumEntries);
	}

	virtual void Free()
	{
		k_PointStorage::Free();
		CUDA_FREE(m_pDeviceData);
	}

	virtual void PrepareForRendering();

	template<bool USE_GLOBAL> CUDA_FUNC_IN Spectrum L_Volume(float a_r, CudaRNG& rng, const Ray& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr)
	{
		Spectrum Tau = Spectrum(0.0f);
		float r2 = a_r * a_r;
		Spectrum L_n = Spectrum(0.0f);
		TraverseGrid(r, m_sStorage.hashMap, tmin, tmax, [&](float minT, float rayT, float maxT, float cellEndT, Vec3u& cell_pos, bool& cancelTraversal)
		{
#ifdef ISCUDA
			int2 beam;
			beam.y = m_sStorage.hashMap.Hash(cell_pos);
			while (beam.y != -1)
			{
				beam = m_pDeviceData[beam.y];
				if (beam.x != -1)
				{
					const volPhoton& ph = m_sStorage(beam.x);
					float l1 = dot(ph.p - r.origin, r.direction) / dot(r.direction, r.direction);
					if (distanceSquared(ph.p, r(l1)) < ph.rad && rayT <= l1 && l1 <= cellEndT)
					{
						float p = vol.p(P, r.direction, wi, rng);
						Spectrum tauToPhoton = (-Tau - vol.tau(r, rayT, l1)).exp();
						L_n += p * ph.phi / (PI * m_uNumEmitted * r_p2) * tauToPhoton;
					}
				}
			}
			float localDist = cellEndT - rayT;
			Spectrum tauD = vol.tau(r, rayT, cellEndT);
			Tau += tauD;
			L_n += vol.Lve(r(rayT + localDist / 2), -1.0f * r.direction) * localDist;
#endif
		});
		Tr = (-Tau).exp();
		return L_n;
	}
};