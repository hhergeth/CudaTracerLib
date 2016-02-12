#pragma once
#include "Beam.h"
#include <Engine/SpatialGrid.h>
#include <Math/half.h>
#include <Math/Compression.h>

namespace CudaTracerLib {

struct PointStorage : public IVolumeEstimator
{
	struct volPhoton
	{
		unsigned int flag_type_pos;
		RGBE phi;
		unsigned short wi;
		half r;
		CUDA_FUNC_IN volPhoton(){}
		CUDA_FUNC_IN volPhoton(const Vec3f& p, const NormalizedT<Vec3f>& w, const Spectrum& ph, const HashGrid_Reg& grid, const Vec3u& cell_idx)
		{
			r = half(0.0f);
			flag_type_pos = grid.EncodePos(p, cell_idx);
			phi = ph.toRGBE();
			wi = NormalizedFloat3ToUchar2(w);
		}
		CUDA_FUNC_IN Vec3f getPos(const HashGrid_Reg& grid, const Vec3u& cell_idx) const
		{
			return grid.DecodePos(flag_type_pos & 0x3fffffff, cell_idx);
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
		CUDA_FUNC_IN float getRad() const
		{
			return r.ToFloat();
		}
		CUDA_FUNC_IN void setRad(float f)
		{
			r = half(f);
		}
		CUDA_FUNC_IN bool getFlag() const
		{
			return (flag_type_pos >> 31) != 0;
		}
		CUDA_FUNC_IN void setFlag()
		{
			flag_type_pos |= 0x80000000;
		}
	};
	SpatialLinkedMap<volPhoton> m_sStorage;
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
		m_sStorage.ResetBuffer();
	}

	virtual void StartNewRendering(const AABB& box)
	{
		m_sStorage.SetSceneDimensions(box);
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
#ifdef __CUDACC__
	CUDA_ONLY_FUNC bool StoreBeam(const Beam& b)
	{
		return false;
	}

	CUDA_ONLY_FUNC bool StorePhoton(const Vec3f& pos, const NormalizedT<Vec3f>& wi, const Spectrum& phi)
	{
		if(!m_sStorage.getHashGrid().getAABB().Contains(pos))
			return false;
		Vec3u cell_idx = m_sStorage.getHashGrid().Transform(pos);
		return m_sStorage.store(cell_idx, volPhoton(pos, wi, phi, m_sStorage.getHashGrid(), cell_idx));
	}

	template<bool USE_GLOBAL> CUDA_FUNC_IN Spectrum L_Volume(float NumEmitted, float a_r, CudaRNG& rng, const NormalizedT<Ray>& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, Spectrum& Tr);
#endif
};

}
