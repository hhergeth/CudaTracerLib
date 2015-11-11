#pragma once
#include <Defines.h>
#include <Math/AABB.h>
#include "PhaseFunction.h"
#include <VirtualFuncType.h>

//Implementation and interface copied from Mitsuba as well as PBRT.

namespace CudaTracerLib {

template<typename T> class Stream;

struct MediumSamplingRecord
{
	float t;
	Vec3f p;
	Vec3f orientation;
	Spectrum transmittance;
	Spectrum sigmaA;
	Spectrum sigmaS;
	float pdfSuccess;
	float pdfSuccessRev;
	float pdfFailure;
};

struct BaseVolumeRegion : public BaseType//, public BaseTypeHelper<5001046>
{
public:
	unsigned int m_uNodeIndex;
	PhaseFunction Func;
	float4x4 WorldToVolume, VolumeToWorld;

	CUDA_FUNC_IN BaseVolumeRegion()
	{
		m_uNodeIndex = UINT_MAX;
	}

	virtual void Update()
	{
		WorldToVolume = VolumeToWorld.inverse();
	}

	CUDA_FUNC_IN bool insideWorld(const Vec3f& p) const
	{
		Vec3f pl = WorldToVolume.TransformPoint(p);
		return pl.x >= 0 && pl.y >= 0 && pl.z >= 0 &&
			pl.x <= 1 && pl.y <= 1 && pl.z <= 1;
	}

	CUDA_DEVICE CUDA_HOST bool IntersectP(const Ray &ray, const float minT, const float maxT, float *t0, float *t1) const;
};

struct HomogeneousVolumeDensity : public BaseVolumeRegion//, public e_DerivedTypeHelper<1>
{
	TYPE_FUNC(1)
public:
	HomogeneousVolumeDensity(){}
	HomogeneousVolumeDensity(const PhaseFunction& func, const float4x4& ToWorld, const float sa, const float ss, float emit)
	{
		BaseVolumeRegion::Func = func;
		VolumeToWorld = ToWorld;
		sig_a = Spectrum(sa);
		sig_s = Spectrum(ss);
		le = Spectrum(emit);
	}

	CUDA_FUNC_IN HomogeneousVolumeDensity(const PhaseFunction& func, const float4x4& ToWorld, const Spectrum& sa, const Spectrum& ss, const Spectrum& emit)
	{
		BaseVolumeRegion::Func = func;
		VolumeToWorld = ToWorld;
		sig_a = sa;
		sig_s = ss;
		le = emit;
	}

	CUDA_FUNC_IN Spectrum sigma_a(const Vec3f& p, const Vec3f& w) const
	{
		return insideWorld(p) ? sig_a : Spectrum(0.0f);
	}

	CUDA_FUNC_IN Spectrum sigma_s(const Vec3f& p, const Vec3f& w) const
	{
		return insideWorld(p) ? sig_s : Spectrum(0.0f);
	}

	CUDA_FUNC_IN Spectrum Lve(const Vec3f& p, const Vec3f& w) const
	{
		return insideWorld(p) ? le : Spectrum(0.0f);
	}

	CUDA_FUNC_IN Spectrum sigma_t(const Vec3f &p, const Vec3f &wo) const
	{
		return insideWorld(p) ? (sig_s + sig_a) : Spectrum(0.0f);
	}

	CUDA_DEVICE CUDA_HOST Spectrum tau(const Ray &ray, const float minT, const float maxT) const;

	CUDA_DEVICE CUDA_HOST bool sampleDistance(const Ray& ray, float minT, float maxT, float sample, MediumSamplingRecord& mRec) const;
public:
	Spectrum sig_a, sig_s, le;
};

struct DenseVolGridBaseType
{
	e_Variable<char> data;
	DenseVolGridBaseType()
	{
	}
	DenseVolGridBaseType(Stream<char>* a_Buffer, Vec3u dim, size_t sizePerElement);
	void InvalidateDeviceData(Stream<char>* a_Buffer);
	template<typename T> CUDA_FUNC_IN e_Variable<T> getVar() const
	{
		return data.As<T>();
	}
};

template<typename T> struct DenseVolGrid : public DenseVolGridBaseType
{
public:
	Vec3u dim;
	Vec3f dimF;
	DenseVolGrid(){}
	DenseVolGrid(Stream<char>* a_Buffer, Vec3u dim)
		: DenseVolGridBaseType(a_Buffer, dim, sizeof(T)), dim(dim)
	{
		dimF = Vec3f(dim.x, dim.y, dim.z);
	}
	void Clear()
	{
		memset(data.host, 0, sizeof(T)* dim.x * dim.y * dim.z);
		cudaMemset(data.device, 0, sizeof(T)* dim.x * dim.y * dim.z);
	}
	void Set(const T& val)
	{
		for (int i = 0; i < dim.x; i++)
			for (int j = 0; j < dim.y; j++)
				for (int k = 0; k < dim.z; k++)
					value(i, j, k) = val;
	}
	CUDA_FUNC_IN unsigned int idx(unsigned int i, unsigned int j, unsigned int k) const
	{
		return min(k, dim.x - 1) + min(j, dim.y - 1) * dim.x + min(i, dim.z - 1) * dim.x * dim.y;
	}
	CUDA_FUNC_IN bool isInBounds(const Vec3u& idx) const
	{
		return idx.x < dim.x && idx.y < dim.y && idx.z < dim.z;
	}
	CUDA_FUNC_IN T& value(unsigned int i, unsigned int j, unsigned int k)
	{
		return getVar<T>()[idx(i, j, k)];
	}
	CUDA_FUNC_IN const T& value(unsigned int i, unsigned int j, unsigned int k) const
	{
		return getVar<T>()[idx(i, j, k)];
	}
	CUDA_FUNC_IN T sampleTrilinear(const Vec3f& vsP) const
	{
		const Vec3f p = vsP - Vec3f(0.5f);
		const Vec3u corner = Vec3u(unsigned int(p.x), unsigned int(p.y), unsigned int(p.z));
		float weight[3];
		T val = T(0);
		Vec3u cl_l = Vec3u(0), cl_h = dim - Vec3u(1);
		for (int i = 0; i < 2; i++)
		{
			unsigned int cur_x = corner.x + i;
			weight[0] = 1 - math::abs(p.x - cur_x);
			for (int j = 0; j < 2; j++)
			{
				unsigned int cur_y = corner.y + j;
				weight[1] = 1 - math::abs(p.y - cur_y);
				for (int k = 0; k < 2; k++)
				{
					unsigned int cur_z = corner.z + k;
					weight[2] = 1 - math::abs(p.z - cur_z);
					val += weight[0] * weight[1] * weight[2] * value(math::clamp(cur_x, cl_l.x, cl_h.x), math::clamp(cur_y, cl_l.y, cl_h.y), math::clamp(cur_z, cl_l.z, cl_h.z));
				}
			}
		}
		return val;
	}
};

struct VolumeGrid : public BaseVolumeRegion//, public e_DerivedTypeHelper<2>
{
	TYPE_FUNC(2)
public:
	VolumeGrid(){}
	VolumeGrid(const PhaseFunction& func, const float4x4& ToWorld, Stream<char>* a_Buffer, Vec3u dim);
	VolumeGrid(const PhaseFunction& func, const float4x4& ToWorld, Stream<char>* a_Buffer, Vec3u dimA, Vec3u dimS, Vec3u dimL);

	CUDA_FUNC_IN Spectrum sigma_a(const Vec3f& p, const Vec3f& w) const
	{
		return sigAMin + (sigAMax - sigAMin) * densityA(p);
	}

	CUDA_FUNC_IN Spectrum sigma_s(const Vec3f& p, const Vec3f& w) const
	{
		return sigSMin + (sigSMax - sigSMin) * densityS(p);
	}

	CUDA_FUNC_IN Spectrum Lve(const Vec3f& p, const Vec3f& w) const
	{
		return leMin + (leMax - leMin) * densityL(p);
	}

	CUDA_FUNC_IN Spectrum sigma_t(const Vec3f &p, const Vec3f &wo) const
	{
		float a, s;
		densityT(p, a, s);
		return sigAMin + (sigAMax - sigAMin) * a + sigSMin + (sigSMax - sigSMin) * s;
	}

	CUDA_DEVICE CUDA_HOST float integrateDensity(const Ray& ray, float minT, float maxT) const;

	CUDA_DEVICE CUDA_HOST bool invertDensityIntegral(const Ray& ray, float minT, float maxT, float desiredDensity,
		float &integratedDensity, float &t, float &densityAtMinT, float &densityAtT) const;

	CUDA_DEVICE CUDA_HOST Spectrum tau(const Ray &ray, const float minT, const float maxT) const;

	CUDA_DEVICE CUDA_HOST bool sampleDistance(const Ray& ray, float minT, float maxT, float sample, MediumSamplingRecord& mRec) const;

	CUDA_FUNC_IN void Voxelize(const Vec3f& p, const DenseVolGrid<float>* V, float& i, float& j, float& k) const
	{
		Vec3f f = tr(p, V->dimF);
		i = f.x;
		j = f.y;
		k = f.z;
	}

	CUDA_FUNC_IN void VoxelToWorld(int i, int j, int k, const DenseVolGrid<float>* V, float& a, float& b, float& c) const
	{
		Vec3f f = Vec3f(float(i) / V->dimF.x, float(j) / V->dimF.y, float(k) / V->dimF.z);
		f = VolumeToWorld.TransformPoint(f);
		a = f.x;
		b = f.y;
		c = f.z;
	}

	virtual void Update();
public:
	Spectrum sigAMin, sigAMax, sigSMin, sigSMax, leMin, leMax;
	DenseVolGrid<float> gridA, gridS, gridL, grid;
	bool singleGrid;
	float m_stepSize;
private:
	CUDA_FUNC_IN Vec3f tr(const Vec3f& p, const Vec3f& dimF) const
	{
		Vec3f csP = WorldToVolume.TransformPoint(p);
		csP = math::clamp01(csP) * dimF;
		return csP;
	}
	CUDA_FUNC_IN float densityA(const Vec3f& p) const
	{
		if (singleGrid)
			return grid.sampleTrilinear(tr(p, grid.dimF));
		else return gridA.sampleTrilinear(tr(p, gridA.dimF));
	}
	CUDA_FUNC_IN float densityS(const Vec3f& p) const
	{
		if (singleGrid)
			return grid.sampleTrilinear(tr(p, grid.dimF));
		else return gridS.sampleTrilinear(tr(p, gridS.dimF));
	}
	CUDA_FUNC_IN float densityL(const Vec3f& p) const
	{
		if (singleGrid)
			return grid.sampleTrilinear(tr(p, grid.dimF));
		else return gridL.sampleTrilinear(tr(p, gridL.dimF));
	}
	CUDA_FUNC_IN float densityT(const Vec3f& p) const
	{
		if (singleGrid)
			return 2 * grid.sampleTrilinear(tr(p, grid.dimF));
		else return gridA.sampleTrilinear(tr(p, gridA.dimF)) + gridS.sampleTrilinear(tr(p, gridS.dimF));
	}
	CUDA_FUNC_IN void densityT(const Vec3f& p, float& a, float& s) const
	{
		if (singleGrid)
			a = s = grid.sampleTrilinear(tr(p, grid.dimF));
		else
		{
			a = gridA.sampleTrilinear(tr(p, gridA.dimF));
			s = gridS.sampleTrilinear(tr(p, gridS.dimF));
		};
	}
};

struct CUDA_ALIGN(16) VolumeRegion : public CudaVirtualAggregate<BaseVolumeRegion, HomogeneousVolumeDensity, VolumeGrid>
{
public:
	CUDA_FUNC_IN AABB WorldBound() const
	{
		return AABB(Vec3f(0), Vec3f(1)).Transform(((BaseVolumeRegion*)Data)->VolumeToWorld);
	}

	CUDA_FUNC_IN bool IntersectP(const Ray &ray, const float minT, const float maxT, float *t0, float *t1) const
	{
		return As()->IntersectP(ray, minT, maxT, t0, t1);
	}

	CALLER(sigma_a)
		CUDA_FUNC_IN Spectrum sigma_a(const Vec3f& p, const Vec3f& w) const
	{
		return sigma_a_Caller<Spectrum>(*this, p, w);
	}

	CALLER(sigma_s)
		CUDA_FUNC_IN Spectrum sigma_s(const Vec3f& p, const Vec3f& w) const
	{
		return sigma_s_Caller<Spectrum>(*this, p, w);
	}

	CALLER(Lve)
		CUDA_FUNC_IN Spectrum Lve(const Vec3f& p, const Vec3f& w) const
	{
		return Lve_Caller<Spectrum>(*this, p, w);
	}

	CALLER(sigma_t)
		CUDA_FUNC_IN Spectrum sigma_t(const Vec3f &p, const Vec3f &wo) const
	{
		return sigma_t_Caller<Spectrum>(*this, p, wo);
	}

	CALLER(tau)
		CUDA_FUNC_IN Spectrum tau(const Ray &ray, float minT, float maxT) const
	{
		return tau_Caller<Spectrum>(*this, ray, minT, maxT);
	}

	CALLER(sampleDistance)
		CUDA_FUNC_IN bool sampleDistance(const Ray& ray, float minT, float maxT, float sample, MediumSamplingRecord& mRec) const
	{
		return sampleDistance_Caller<bool>(*this, ray, minT, maxT, sample, mRec);
	}
};

struct KernelAggregateVolume
{
	enum{ MAX_VOL_COUNT = 16 };
public:
	unsigned int m_uVolumeCount;
	VolumeRegion m_pVolumes[MAX_VOL_COUNT];
	AABB box;
public:
	CUDA_FUNC_IN KernelAggregateVolume(){}
	KernelAggregateVolume(Stream<VolumeRegion>* D, bool devicePointer = true);

	///Calculates the intersection of the ray with the bound of the volume
	CUDA_DEVICE CUDA_HOST bool IntersectP(const Ray &ray, float minT, float maxT, float *t0, float *t1) const;

	///The probability that light is abosrbed per unit distance
	CUDA_DEVICE CUDA_HOST Spectrum sigma_a(const Vec3f& p, const Vec3f& w) const;

	///The probability that light is scattered per unit distance
	CUDA_DEVICE CUDA_HOST Spectrum sigma_s(const Vec3f& p, const Vec3f& w) const;

	CUDA_DEVICE CUDA_HOST Spectrum Lve(const Vec3f& p, const Vec3f& w) const;

	///Combined sigmas
	CUDA_DEVICE CUDA_HOST Spectrum sigma_t(const Vec3f &p, const Vec3f &wo) const;

	///Calculates the volumes optical thickness along a ray in the volumes bounds
	CUDA_DEVICE CUDA_HOST Spectrum tau(const Ray &ray, float minT, float maxT) const;

	CUDA_DEVICE CUDA_HOST float Sample(const Vec3f& p, const Vec3f& wo, CudaRNG& rng, Vec3f* wi);

	CUDA_DEVICE CUDA_HOST float p(const Vec3f& p, const Vec3f& wo, const Vec3f& wi, CudaRNG& rng);

	CUDA_DEVICE CUDA_HOST bool sampleDistance(const Ray& ray, float minT, float maxT, CudaRNG& rng, MediumSamplingRecord& mRec) const;

	CUDA_FUNC_IN bool HasVolumes()
	{
		return m_uVolumeCount > 0;
	}

	CUDA_FUNC_IN bool IsInVolume(const Vec3f& p) const
	{
		for (unsigned int i = 0; i < m_uVolumeCount; i++)
			if (m_pVolumes[i].As()->insideWorld(p))
				return true;
		return false;
	}
};

}