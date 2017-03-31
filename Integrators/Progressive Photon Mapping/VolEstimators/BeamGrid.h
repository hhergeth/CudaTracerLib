#pragma once
#include "PointStorage.h"
#include <functional>

namespace CudaTracerLib {

struct BeamGrid : public PointStorage
{
	typedef std::function<float()> kNN_clb_t;

	struct entry
	{
		int i;

		entry()
		{

		}

		CUDA_FUNC_IN entry(int i)
			: i(i)
		{

		}

		CUDA_FUNC_IN bool getFlag() const
		{
			return (i >> 31) == 1;
		}

		CUDA_FUNC_IN void setFlag()
		{
			i |= 1 << 31;
		}

		CUDA_FUNC_IN int getIndex() const
		{
			return i & 0x7fffffff;
		}
	};

	SpatialGridList_Linked<entry> m_sBeamGridStorage;

	kNN_clb_t m_kNNClb;

	CUDA_FUNC_IN static constexpr int DIM()
	{
		return 2;
	}

	BeamGrid(unsigned int gridDim, unsigned int numPhotons, kNN_clb_t clb, int N = 20)
		: PointStorage(gridDim, numPhotons, m_sBeamGridStorage), m_kNNClb(clb), m_sBeamGridStorage(Vec3u(gridDim), gridDim * gridDim * gridDim * (1 + N))
	{
		if (!m_kNNClb)
			throw std::runtime_error("Please provide a kNN callback!");
	}

	virtual void Free()
	{
		PointStorage::Free();
		m_sBeamGridStorage.Free();
	}

	virtual void StartNewPass(DynamicScene* scene);

	virtual void StartNewRendering(const AABB& box)
	{
		PointStorage::StartNewRendering(box);
		m_sBeamGridStorage.SetGridDimensions(box);
	}

	virtual size_t getSize() const
	{
		return sizeof(*this);
	}

	CTL_EXPORT virtual void PrepareForRendering();

	virtual void PrintStatus(std::vector<std::string>& a_Buf) const
	{
		PointStorage::PrintStatus(a_Buf);
		a_Buf.push_back(format("%.2f%% Beam indices", (float)m_sBeamGridStorage.getNumStoredEntries() / m_sBeamGridStorage.getNumEntries() * 100));
	}

	template<bool USE_GLOBAL> CUDA_FUNC_IN Spectrum L_Volume(float rad, float NumEmitted, const NormalizedT<Ray>& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr, float& pl_est)
	{
		Spectrum Tau = Spectrum(0.0f);
		Spectrum L_n = Spectrum(0.0f);
		TraverseGridRay(r, tmin, tmax, m_sStorage, [&](float minT, float rayT, float maxT, float cellEndT, const Vec3u& cell_pos, bool& cancelTraversal)
		{
			m_sBeamGridStorage.ForAllCellEntries(cell_pos, [&](unsigned int, entry beam_idx)
			{
				const auto& ph = m_sStorage(beam_idx.getIndex());
				Vec3f ph_pos = ph.getPos(m_sStorage.getHashGrid(), cell_pos);
				float ph_rad1 = ph.getRad1(), ph_rad2 = math::sqr(ph_rad1);
				float l1 = dot(ph_pos - r.ori(), r.dir());
				float isectRadSqr = distanceSquared(ph_pos, r(l1));
				if (isectRadSqr < ph_rad2 && rayT <= l1 && l1 <= cellEndT)
				{
					//transmittance from camera vertex along ray to query point
					Spectrum tauToPhoton = (-Tau - vol.tau(r, rayT, l1)).exp();
					PhaseFunctionSamplingRecord pRec(-r.dir(), ph.getWi());
					float p = vol.p(ph_pos, pRec);
					L_n += p * ph.getL() / NumEmitted * tauToPhoton * Kernel::k<2>(math::sqrt(isectRadSqr), ph_rad1);
				}
				/*float t1, t2;
				if (sphere_line_intersection(ph_pos, ph_rad2, r, t1, t2))
				{
				float t = (t1 + t2) / 2;
				auto b = r(t);
				float dist = distance(b, ph_pos);
				auto o_s = vol.sigma_s(b, r.dir()), o_a = vol.sigma_a(b, r.dir()), o_t = Spectrum(o_s + o_a);
				if (dist < ph_rad1 && rayT <= t && t <= cellEndT)
				{
				PhaseFunctionSamplingRecord pRec(-r.dir(), ph.getWi());
				float p = vol.p(b, pRec);

				//auto T1 = (-vol.tau(r, 0, t1)).exp(), T2 = (-vol.tau(r, 0, t2)).exp(),
				//	 ta = (t2 - t1) * (T1 + 0.5 * (T2 - T1));
				//L_n += p * ph.getL() / NumEmitted * Kernel::k<3>(dist, ph_rad1) * ta;
				auto Tr_c = (-vol.tau(r, 0, t)).exp();
				L_n += p * ph.getL() / NumEmitted * Kernel::k<3>(dist, ph_rad1) * Tr_c * (t2 - t1);
				}
				}*/
			});
			Tau += vol.tau(r, rayT, cellEndT);
			float localDist = cellEndT - rayT;
			L_n += vol.Lve(r(rayT + localDist / 2), -r.dir()) * localDist;
		});
		Tr = (-Tau).exp();

		return L_n;
	}
};

}