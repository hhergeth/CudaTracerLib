#pragma once

#include "..\Math\AABB.h"
#include "e_Buffer.h"
#include "e_PhaseFunction.h"

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

CUDA_FUNC_IN bool sampleDistanceHomogenous(const Ray& ray, float minT, float maxT, float rand, MediumSamplingRecord& mRec, const Spectrum& sig_a, const Spectrum& sig_s)
{
	float m_mediumSamplingWeight = -1;
	Spectrum sig_t = sig_a + sig_s, albedo = sig_s / sig_t;
	for (int i = 0; i < 2; i++)
	if (albedo[i] > m_mediumSamplingWeight && sig_t[i] != 0)
		m_mediumSamplingWeight = albedo[i];
	if (m_mediumSamplingWeight > 0)
		m_mediumSamplingWeight = max(m_mediumSamplingWeight, 0.5f);
	float sampledDistance;
	int channel = int(rand * SPECTRUM_SAMPLES);
	rand = (rand - channel * 1.0f / SPECTRUM_SAMPLES) * SPECTRUM_SAMPLES;
	if (rand < m_mediumSamplingWeight)
	{
		rand /= m_mediumSamplingWeight;
		float samplingDensity = sig_t[channel];
		sampledDistance = -logf(1 - rand) / samplingDensity;
	}
	bool success = true;
	if (sampledDistance < maxT - minT)
	{
		mRec.t = minT + sampledDistance;
		mRec.p = ray(mRec.t);
		mRec.sigmaA = sig_a;
		mRec.sigmaS = sig_s;
		if (mRec.p == ray.origin)
			success = false;
	}
	else
	{
		sampledDistance = maxT - minT;
		success = false;
	}

	mRec.pdfFailure = 0;
	mRec.pdfSuccess = 0;
	for (int i = 0; i < SPECTRUM_SAMPLES; i++)
	{
		float t = math::exp(-sig_t[i] * sampledDistance);
		mRec.pdfFailure += t;
		mRec.pdfSuccess += sig_t[i] * t;
	}
	mRec.pdfFailure /= SPECTRUM_SAMPLES;
	mRec.pdfSuccess /= SPECTRUM_SAMPLES;
	mRec.transmittance = (sig_t * (-sampledDistance)).exp();
	mRec.pdfSuccessRev = mRec.pdfSuccess = mRec.pdfSuccess * m_mediumSamplingWeight;
	mRec.pdfFailure = m_mediumSamplingWeight * mRec.pdfFailure + (1 - m_mediumSamplingWeight);
	if (mRec.transmittance.max() < 1e-10f)
		mRec.transmittance = Spectrum(0.0f);

	return success;
}

struct e_BaseVolumeRegion : public e_BaseType//, public e_BaseTypeHelper<5001046>
{
public:
	unsigned int m_uNodeIndex;
	AABB Box;
	e_PhaseFunction Func;

	e_BaseVolumeRegion()
	{
		m_uNodeIndex = 0xffffffff;
	}

	CUDA_FUNC_IN bool isInVolume(unsigned int a_NodeIndex)
	{
		return m_uNodeIndex == 0xffffffff || m_uNodeIndex == a_NodeIndex;
	}

	CUDA_FUNC_IN bool inside(const Vec3f& p) const
	{
		return Box.Contains(p);
	}
};

struct e_HomogeneousVolumeDensity : public e_BaseVolumeRegion//, public e_DerivedTypeHelper<1>
{
	TYPE_FUNC(1)
public:
	e_HomogeneousVolumeDensity(){}
	e_HomogeneousVolumeDensity(const float sa, const float ss, const e_PhaseFunction& func, float emit, const AABB& box)
	{
		e_BaseVolumeRegion::Box = box;
		e_BaseVolumeRegion::Func = func;
        WorldToVolume = float4x4::Identity();
        sig_a = Spectrum(sa);
        sig_s = Spectrum(ss);
        le = Spectrum(emit);
	}

	e_HomogeneousVolumeDensity(const Spectrum& sa, const Spectrum& ss, const e_PhaseFunction& func, const Spectrum& emit, const AABB& box, const float4x4& v2w)
	{
		e_BaseVolumeRegion::Box = box;
		e_BaseVolumeRegion::Func = func;
        WorldToVolume = v2w.inverse();
        sig_a = sa;
        sig_s = ss;
        le = emit;
	}

    CUDA_DEVICE CUDA_HOST bool IntersectP(const Ray &ray, const float minT, const float maxT, float *t0, float *t1) const;

	CUDA_FUNC_IN Spectrum sigma_a(const Vec3f& p, const Vec3f& w) const
	{
		return inside(p) ? sig_a : Spectrum(0.0f);
	}

	CUDA_FUNC_IN Spectrum sigma_s(const Vec3f& p, const Vec3f& w) const
	{
		return inside(p) ? sig_s : Spectrum(0.0f);
	}

	CUDA_FUNC_IN Spectrum Lve(const Vec3f& p, const Vec3f& w) const
	{
		return inside(p) ? le : Spectrum(0.0f);
	}

	CUDA_FUNC_IN Spectrum sigma_t(const Vec3f &p, const Vec3f &wo) const
	{
		return inside(p) ? (sig_s + sig_a) : Spectrum(0.0f);
	}

    CUDA_DEVICE CUDA_HOST Spectrum tau(const Ray &ray, const float minT, const float maxT) const;

	CUDA_DEVICE CUDA_HOST bool sampleDistance(const Ray& ray, float minT, float maxT, float sample, MediumSamplingRecord& mRec) const;
public:
	Spectrum sig_a, sig_s, le;
	float4x4 WorldToVolume;
};

template<typename T> struct e_DenseVolGrid
{
private:
	e_Variable<T> data;
	e_StreamReference(char) streamRef;
public:
	Vec3u dim;
	Vec3f dimF;
	e_DenseVolGrid(){}
	e_DenseVolGrid(e_Stream<char>* a_Buffer, Vec3u dim)
		: dim(dim)
	{
		dimF = Vec3f(dim.x, dim.y, dim.z);
		streamRef = a_Buffer->malloc(dim.x * dim.y * dim.z * sizeof(T));
		data = streamRef.AsVar<T>();
	}
	void Clear()
	{
		memset(data.host, 0, sizeof(T)* dim.x * dim.y * dim.z);
		cudaMemset(data.device, 0, sizeof(T)* dim.x * dim.y * dim.z);
	}
	void InvalidateDeviceData()
	{
		streamRef.Invalidate();
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
		return k + j * dim.x + i * dim.x * dim.y;
	}
	CUDA_FUNC_IN bool isInBounds(const Vec3u& idx) const
	{
		return idx.x < dim.x && idx.y < dim.y && idx.z < dim.z;
	}
	CUDA_FUNC_IN T& value(unsigned int i, unsigned int j, unsigned int k)
	{
		return data[idx(i, j, k)];
	}
	CUDA_FUNC_IN const T& value(unsigned int i, unsigned int j, unsigned int k) const
	{
		return data[idx(i, j, k)];
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

struct e_VolumeGrid : public e_BaseVolumeRegion//, public e_DerivedTypeHelper<2>
{
	TYPE_FUNC(2)
public:
	e_VolumeGrid(){}
	e_VolumeGrid(const e_PhaseFunction& func, const float4x4 worldToVol, e_Stream<char>* a_Buffer, Vec3u dim);
	e_VolumeGrid(const e_PhaseFunction& func, const float4x4 worldToVol, e_Stream<char>* a_Buffer, Vec3u dimA, Vec3u dimS, Vec3u dimL);

	CUDA_DEVICE CUDA_HOST bool IntersectP(const Ray &ray, const float minT, const float maxT, float *t0, float *t1) const;

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

	CUDA_FUNC_IN void Voxelize(const Vec3f& p, const e_DenseVolGrid<float>* V, float& i, float& j, float& k) const
	{
		Vec3f f = tr(p, V->dimF);
		i = f.x;
		j = f.y;
		k = f.z;
	}

	CUDA_FUNC_IN void VoxelToWorld(int i, int j, int k, const e_DenseVolGrid<float>* V, float& a, float& b, float& c) const
	{
		Vec3f f = Vec3f(float(i) / V->dimF.x, float(j) / V->dimF.y, float(k) / V->dimF.z);
		f = VolumeToWorld.TransformPoint(f);
		a = f.x;
		b = f.y;
		c = f.z;
	}

	virtual void Update();
public:
	float4x4 WorldToVolume, VolumeToWorld;
	Spectrum sigAMin, sigAMax, sigSMin, sigSMax, leMin, leMax;
	e_DenseVolGrid<float> gridA, gridS, gridL, grid;
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

struct CUDA_ALIGN(16) e_VolumeRegion : public CudaVirtualAggregate<e_BaseVolumeRegion, e_HomogeneousVolumeDensity, e_VolumeGrid>
{
public:
	CUDA_FUNC_IN e_VolumeRegion()
	{
		type = 0;
	}

	CUDA_FUNC_IN e_BaseVolumeRegion* BaseRegion()
	{
		return (e_BaseVolumeRegion*)Data;
	}

	CUDA_FUNC_IN AABB WorldBound()
	{
		return ((e_BaseVolumeRegion*)Data)->Box;
	}

	CUDA_FUNC_IN bool IntersectP(const Ray &ray, const float minT, const float maxT, float *t0, float *t1) const
	{
		CALL_FUNC2(e_HomogeneousVolumeDensity, e_VolumeGrid, IntersectP(ray, minT, maxT, t0, t1))
		return false;
	}

	CUDA_FUNC_IN Spectrum sigma_a(const Vec3f& p, const Vec3f& w) const
	{
		CALL_FUNC2(e_HomogeneousVolumeDensity, e_VolumeGrid, sigma_a(p, w))
		return 0.0f;
	}

	CUDA_FUNC_IN Spectrum sigma_s(const Vec3f& p, const Vec3f& w) const
	{
		CALL_FUNC2(e_HomogeneousVolumeDensity, e_VolumeGrid, sigma_s(p, w))
		return 0.0f;
	}

	CUDA_FUNC_IN Spectrum Lve(const Vec3f& p, const Vec3f& w) const
	{
		CALL_FUNC2(e_HomogeneousVolumeDensity, e_VolumeGrid, Lve(p, w))
		return 0.0f;
	}

	CUDA_FUNC_IN Spectrum sigma_t(const Vec3f &p, const Vec3f &wo) const
	{
		CALL_FUNC2(e_HomogeneousVolumeDensity, e_VolumeGrid, sigma_t(p, wo))
		return 0.0f;
	}

    CUDA_FUNC_IN Spectrum tau(const Ray &ray, float minT, float maxT) const
	{
		CALL_FUNC2(e_HomogeneousVolumeDensity, e_VolumeGrid, tau(ray, minT, maxT))
		return 0.0f;
	}

	CUDA_FUNC_IN bool sampleDistance(const Ray& ray, float minT, float maxT, float sample, MediumSamplingRecord& mRec) const
	{
		CALL_FUNC2(e_HomogeneousVolumeDensity, e_VolumeGrid, sampleDistance(ray, minT, maxT, sample, mRec))
		return false;
	}
};

struct e_KernelAggregateVolume
{
public:
	unsigned int m_uVolumeCount;
	e_VolumeRegion* m_pVolumes;
	AABB box;
public:
	CUDA_FUNC_IN e_KernelAggregateVolume()
	{

	}
	e_KernelAggregateVolume(e_Stream<e_VolumeRegion>* D, bool devicePointer = true)
	{
		m_uVolumeCount = D->UsedElements().getLength();
		m_pVolumes = D->getKernelData(devicePointer).Data;
		box = AABB::Identity();
		for(unsigned int i = 0; i < m_uVolumeCount; i++)
			box.Enlarge(D->operator()(i)->WorldBound());
	}

	///Calculates the intersection of the ray with the bound of the volume
	CUDA_DEVICE CUDA_HOST bool IntersectP(const Ray &ray, float minT, float maxT, float *t0, float *t1, unsigned int a_NodeIndex = 0xffffffff) const;

	///The probability that light is abosrbed per unit distance
	CUDA_DEVICE CUDA_HOST Spectrum sigma_a(const Vec3f& p, const Vec3f& w, unsigned int a_NodeIndex = 0xffffffff) const;

	///The probability that light is scattered per unit distance
	CUDA_DEVICE CUDA_HOST Spectrum sigma_s(const Vec3f& p, const Vec3f& w, unsigned int a_NodeIndex = 0xffffffff) const;

	CUDA_DEVICE CUDA_HOST Spectrum Lve(const Vec3f& p, const Vec3f& w, unsigned int a_NodeIndex = 0xffffffff) const;

	///Combined sigmas
	CUDA_DEVICE CUDA_HOST Spectrum sigma_t(const Vec3f &p, const Vec3f &wo, unsigned int a_NodeIndex = 0xffffffff) const;

	///Calculates the volumes optical thickness along a ray in the volumes bounds
	CUDA_DEVICE CUDA_HOST Spectrum tau(const Ray &ray, float minT, float maxT, unsigned int a_NodeIndex = 0xffffffff) const;

	CUDA_DEVICE CUDA_HOST float Sample(const Vec3f& p, const Vec3f& wo, CudaRNG& rng, Vec3f* wi, unsigned int a_NodeIndex = 0xffffffff);

	CUDA_DEVICE CUDA_HOST float p(const Vec3f& p, const Vec3f& wo, const Vec3f& wi, CudaRNG& rng, unsigned int a_NodeIndex = 0xffffffff);

	CUDA_DEVICE CUDA_HOST bool sampleDistance(const Ray& ray, float minT, float maxT, CudaRNG& rng, MediumSamplingRecord& mRec, unsigned int a_NodeIndex = 0xffffffff) const;

	CUDA_FUNC_IN bool HasVolumes()
	{
		return m_uVolumeCount > 0;
	}

	CUDA_FUNC_IN bool IsInVolume(const Vec3f& p, unsigned int a_NodeIndex = 0xffffffff) const
	{
		for (unsigned int i = 0; i < m_uVolumeCount; i++)
			if (m_pVolumes[i].BaseRegion()->Box.Contains(p))
				return true;
		return false;
	}
};