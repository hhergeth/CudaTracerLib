#pragma once
#include "k_Beam.h"
#include <Engine/e_SpatialGrid.h>

struct k_BeamBeamGrid : public IVolumeEstimator
{
	e_SpatialLinkedMap<int> m_sStorage;

	k_Beam* m_pDeviceBeams;
	unsigned int m_uBeamIdx;
	unsigned int m_uBeamLength;

	unsigned int m_uNumEmitted;
	float m_fCurrentRadiusVol;

	k_BeamBeamGrid(unsigned int gridDim, unsigned int numBeams, int N = 100)
		: m_sStorage(gridDim, gridDim * gridDim * gridDim * N), m_uBeamLength(numBeams)
	{
		CUDA_MALLOC(&m_pDeviceBeams, sizeof(k_Beam) * m_uBeamLength);
	}

	virtual void Free()
	{
		m_sStorage.Free();
		CUDA_FREE(m_pDeviceBeams);
	}

	virtual void StartNewPass(const IRadiusProvider* radProvider, e_DynamicScene* scene)
	{
		m_fCurrentRadiusVol = radProvider->getCurrentRadius(1);
		m_uNumEmitted = 0;
		m_uBeamIdx = 0;
		m_sStorage.ResetBuffer();
	}

	virtual void StartNewRendering(const AABB& box, float a_InitRadius)
	{
		m_sStorage.SetSceneDimensions(box, a_InitRadius);
	}

	virtual bool isFull() const
	{
		return m_uBeamIdx >= m_uBeamLength;
	}

	virtual void PrintStatus(std::vector<std::string>& a_Buf) const
	{
		a_Buf.push_back(format("%.2f%% Beam grid indices", (float)m_sStorage.deviceDataIdx / m_sStorage.numData * 100));
		a_Buf.push_back(format("%.2f%% Beams", (float)m_uBeamIdx / m_uBeamLength * 100));
	}

	virtual size_t getSize() const
	{
		return sizeof(*this);
	}

	virtual void PrepareForRendering()
	{

	}

	CUDA_ONLY_FUNC void StoreBeam(const k_Beam& b, bool firstStore)
	{
		unsigned int beam_idx = atomicInc(&m_uBeamIdx, (unsigned int)-1);
		if (beam_idx < m_uBeamLength)
		{
			m_pDeviceBeams[beam_idx] = b;
			
			struct hashPos
			{
				unsigned long long flags;
				CUDA_FUNC_IN hashPos() : flags(0) {}
				CUDA_FUNC_IN int idx(const Vec3u& center, const Vec3u& pos)
				{
					Vec3u d = pos - center - Vec3u(7);
					if (d.x > pos.x || d.y > pos.y || d.z > pos.z)
						return -1;
					return d.z * 49 + d.y * 7 + d.x;
				}
				CUDA_FUNC_IN bool isSet(const Vec3u& center, const Vec3u& pos)
				{
					int i = idx(center, pos);
					return i == -1 ? false : (flags >> i) & 1;
				}
				CUDA_FUNC_IN void set(const Vec3u& center, const Vec3u& pos)
				{
					int i = idx(center, pos);
					if (i != -1)
						flags |= 1 << i;
				}
			};

			const int N_L = 100;
			unsigned int cell_list[N_L];
			int i_L = 0;
			unsigned int M = 0xffffffff;
			//Vec3u lastMin(M), lastMax(M);
			hashPos H1, H2;
#ifdef ISCUDA
			TraverseGrid(Ray(b.pos, b.dir), m_sStorage.hashMap, 0, b.t, [&](float minT, float rayT, float maxT, float cellEndT, Vec3u& cell_pos, bool& cancelTraversal)
			{
				H1 = H2;
				Frame f_ray(b.dir);
				Vec3f disk_a = b.pos + b.dir * rayT + f_ray.toWorld(Vec3f(-1, -1, 0)) * m_fCurrentRadiusVol, disk_b = b.pos + b.dir * rayT + f_ray.toWorld(Vec3f(1, 1, 0)) * m_fCurrentRadiusVol;
				Vec3f min_disk = min(disk_a, disk_b), max_disk = max(disk_a, disk_b);
				m_sStorage.ForAllCells(min_disk, max_disk, [&](const Vec3u& cell_idx)
				{
					//if (lastMin.x <= ax && ax <= lastMax.x && lastMin.y <= ay && ay <= lastMax.y && lastMin.z <= az && az <= lastMax.z)
					//	continue;

					unsigned int grid_idx = m_sStorage.hashMap.Hash(cell_idx);
					bool found = false;
					for (int i = 0; i < i_L; i++)
						if (cell_list[i] == grid_idx)
						{
							found = true;
							break;
						}
					if (found)
						return;
					if (i_L < N_L)
						cell_list[i_L++] = grid_idx;
					m_sStorage.store(cell_idx, beam_idx);
				});
				if (i_L == N_L)
					printf("cell_list full\n");
				i_L = 0;
				//lastMin = min_cell;
				//lastMax = max_cell;
			});
			//printf("i_L = %d   ", i_L);
#endif
			if (firstStore)
				atomicInc(&m_uNumEmitted, (unsigned int)-1);
		}
	}

	CUDA_ONLY_FUNC void StorePhoton(const Vec3f& pos, const Vec3f& wi, const Spectrum& phi, bool firstStore)
	{

	}

	template<bool USE_GLOBAL> CUDA_FUNC_IN Spectrum L_Volume(float a_r, CudaRNG& rng, const Ray& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr)
	{
		Spectrum L_n = Spectrum(0.0f), Tau = Spectrum(0.0f);
		TraverseGrid(r, m_sStorage.hashMap, tmin, tmax, [&](float minT, float rayT, float maxT, float cellEndT, Vec3u& cell_pos, bool& cancelTraversal)
		{
			m_sStorage.ForAll(cell_pos, [&](unsigned int ABC, int beam_idx)
			{
				k_Beam B = m_pDeviceBeams[beam_idx];
				float t1, t2;
				skew_lines(r, Ray(B.pos, B.dir), t1, t2);
				Vec3f p_b = B.pos + t2 * B.dir, p_c = r.origin + t1 * r.direction;//m_sStorage.hashMap.Transform(p_b) == cell_pos && 
				if (t1 > 0 && t2 > 0 && t2 < B.t && t1 < cellEndT && distanceSquared(p_c, p_b) < a_r * a_r && m_sStorage.hashMap.Transform(p_c) == cell_pos)
				{
					Spectrum tauC = Tau + vol.tau(r, rayT, t1);
					Spectrum tauP = vol.tau(Ray(B.pos, B.dir), 0, t2);
					float p = vol.p(p_b, (p_c - p_b).normalized(), -B.dir, rng);
					float sin_theta_b = math::sqrt(1 - math::sqr(dot(r.direction, B.dir) / (r.direction.length() * B.dir.length())));
					L_n += 1.0f / (m_uNumEmitted * a_r) * p * B.Phi * (-tauC - tauP).exp() / sin_theta_b;
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