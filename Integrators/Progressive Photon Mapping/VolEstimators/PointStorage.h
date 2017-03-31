#pragma once
#include "Beam.h"
#include <Engine/SpatialStructures/Grid/SpatialGridList.h>
#include <Math/half.h>
#include <Math/Compression.h>

namespace CudaTracerLib {

struct VolumetricPhoton
{
private:
	unsigned int flag_type_pos_ll;
	RGBE phi;
	unsigned short wi;
	half r;
	Vec3f Pos;
public:
	CUDA_FUNC_IN VolumetricPhoton() {}
	CUDA_FUNC_IN VolumetricPhoton(const Vec3f& p, const NormalizedT<Vec3f>& w, const Spectrum& ph)
	{
		Pos = p;
		r = half(0.0f);
		flag_type_pos_ll = 0;// EncodePos<4, decltype(flag_type_pos_ll)>(grid.getAABB(), p);

		phi = ph.toRGBE();
		wi = NormalizedFloat3ToUchar2(w);
	}
	CUDA_FUNC_IN Vec3f getPos(const HashGrid_Reg& grid, const Vec3u& cell_idx) const
	{
		return Pos;
	}
	CUDA_FUNC_IN Vec3f getPos()
	{
		return Pos;
	}
	CUDA_FUNC_IN NormalizedT<Vec3f> getWi() const
	{
		return Uchar2ToNormalizedFloat3(wi);
	}
	CUDA_FUNC_IN Spectrum getL() const
	{
		Spectrum s;
		s.fromRGBE(phi);
		return s;
	}
	CUDA_FUNC_IN float getRad1() const
	{
		return r.ToFloat();
	}
	CUDA_FUNC_IN void setRad1(float f)
	{
		r = half(f);
	}
	CUDA_FUNC_IN bool getFlag() const
	{
		return (flag_type_pos_ll & 1) != 0;
	}
	CUDA_FUNC_IN void setFlag()
	{
		flag_type_pos_ll |= 1;
	}
};

typedef VolumetricPhoton _VolumetricPhoton;

struct PointStorage : public IVolumeEstimator
{
protected:
	template<typename... ARGS> PointStorage(unsigned int gridDim, unsigned int numPhotons, ARGS&... args)
		: IVolumeEstimator(m_sStorage, args...), m_sStorage(Vec3u(gridDim), numPhotons)
	{

	}
public:
	SpatialGridList_Linked<_VolumetricPhoton> m_sStorage;

	CUDA_FUNC_IN static constexpr int DIM()
	{
		return 3;
	}

	PointStorage(unsigned int gridDim, unsigned int numPhotons)
		: IVolumeEstimator(m_sStorage), m_sStorage(Vec3u(gridDim), numPhotons)
	{

	}

	virtual void Free()
	{
		m_sStorage.Free();
	}

	CUDA_FUNC_IN _VolumetricPhoton operator()(unsigned int idx)
	{
		return m_sStorage.operator()(idx);
	}

	virtual void StartNewPass(DynamicScene* scene);

	virtual void StartNewRendering(const AABB& box)
	{
		m_sStorage.SetGridDimensions(box);
	}

	CUDA_FUNC_IN bool isFullK() const
	{
		return m_sStorage.isFull();
	}

	virtual bool isFull() const
	{
		return isFullK();
	}

	virtual void getStatusInfo(size_t& length, size_t& count) const
	{
		length = m_sStorage.getNumEntries();
		count = m_sStorage.getNumStoredEntries();
	}

	CUDA_FUNC_IN unsigned int getNumEntries() const
	{
		return m_sStorage.getNumEntries();
	}

	virtual size_t getSize() const
	{
		return sizeof(*this);
	}

	virtual void PrintStatus(std::vector<std::string>& a_Buf) const
	{
		a_Buf.push_back(format("%.2f%% Vol Photons", m_sStorage.getNumStoredEntries() / (float)m_sStorage.getNumEntries() * 100));
	}

	virtual void PrepareForRendering()
	{
		m_sStorage.PrepareForUse();
	}

	template<typename BEAM> CUDA_ONLY_FUNC unsigned int StoreBeam(const BEAM& b)
	{
		return 0xffffffff;
	}

	template<typename PHOTON> CUDA_ONLY_FUNC unsigned int StorePhoton(const PHOTON& ph, const Vec3f& pos)
	{
		if(!m_sStorage.getHashGrid().getAABB().Contains(pos))
			return 0xffffffff;
		Vec3u cell_idx = m_sStorage.getHashGrid().Transform(pos);
		return m_sStorage.Store(cell_idx, ph);
	}

	template<bool USE_GLOBAL> CUDA_FUNC_IN Spectrum L_Volume(float rad, float NumEmitted, const NormalizedT<Ray>& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr, float& pl_est)
	{
		Spectrum Tau = Spectrum(0.0f);
		Spectrum L_n = Spectrum(0.0f);
		float a, b;
		if (!m_sStorage.getHashGrid().getAABB().Intersect(r, &a, &b))
			return L_n;//that would be dumb
		float minT = a = math::clamp(a, tmin, tmax);
		b = math::clamp(b, tmin, tmax);
		float d = 2.0f * rad;
		float pl = 0, num_est;
		while (a < b)
		{
			float t = a + d / 2.0f;
			Vec3f x = r(t);
			Spectrum L_i(0.0f);
			m_sStorage.ForAll(x - Vec3f(rad), x + Vec3f(rad), [&](const Vec3u& cell_idx, unsigned int p_idx, const _VolumetricPhoton& ph)
			{
				Vec3f ph_pos = ph.getPos(m_sStorage.getHashGrid(), cell_idx);
				auto dist2 = distanceSquared(ph_pos, x);
				if (dist2 < math::sqr(rad))
				{
					PhaseFunctionSamplingRecord pRec(-r.dir(), ph.getWi());
					float p = vol.p(x, pRec);
					float k = Kernel::k<3>(math::sqrt(dist2), rad);
					L_i += p * ph.getL() / NumEmitted * k;
					pl += k;
				}
			});
			L_n += (-Tau - vol.tau(r, a, t)).exp() * L_i * d;
			Tau += vol.tau(r, a, a + d);
			L_n += vol.Lve(x, -r.dir()) * d;
			a += d;
			num_est++;
		}
		Tr = (-Tau).exp();
		pl_est += pl / num_est;
		return L_n;
	}
};

}
