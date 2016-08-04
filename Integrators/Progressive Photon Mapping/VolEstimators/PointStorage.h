#pragma once
#include "Beam.h"
#include <Engine/SpatialGrid.h>
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
	SpatialLinkedMap<_VolumetricPhoton> m_sStorage;
	
	float m_fCurrentRadiusVol;

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

	virtual void StartNewPass(const IRadiusProvider* radProvider, DynamicScene* scene);

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

	template<bool USE_GLOBAL> CUDA_FUNC_IN Spectrum L_Volume(float NumEmitted, unsigned int numIteration, float kToFind, const NormalizedT<Ray>& r, float tmin, float tmax, const VolHelper<USE_GLOBAL>& vol, VolumeModel& model, PPM_Radius_Type radType, Spectrum& Tr);

	CUDA_FUNC_IN void Compute_kNN_radii(float numEmitted, float rad, float kToFind, const NormalizedT<Ray>& r, float tmin, float tmax, VolumeModel& model);
};

}
