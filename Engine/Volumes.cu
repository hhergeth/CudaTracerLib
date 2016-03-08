#include "Buffer.h"
#include "Volumes.h"
#include <Base/CudaRandom.h>
#include "Samples.h"
#include "Grid.h"
#include <Math/Sampling.h>

namespace CudaTracerLib {

bool BaseVolumeRegion::IntersectP(const Ray &ray, const float minT, const float maxT, float *t0, float *t1) const
{
	Ray r = ray * WorldToVolume;
	bool b = AABB(Vec3f(0), Vec3f(1)).Intersect(r, t0, t1);
	if(b)
	{
		*t0 = math::clamp(*t0, minT, maxT);
		*t1 = math::clamp(*t1, minT, maxT);
	}
	return b && *t1 > *t0 && *t1 > 0;
}

Spectrum HomogeneousVolumeDensity::tau(const Ray &ray, const float minT, const float maxT) const
{
	float t0, t1;
	if (!IntersectP(ray, minT, maxT, &t0, &t1))
	{
		return Spectrum(0.0f);
	}
	return length(ray(t0) - ray(t1)) * (sig_a + sig_s);
}

bool HomogeneousVolumeDensity::sampleDistance(const Ray& ray, float minT, float maxT, float rand, MediumSamplingRecord& mRec) const
{
	float m_mediumSamplingWeight = -1;
	Spectrum sig_t = sig_a + sig_s, albedo = sig_s / sig_t;
	for (int i = 0; i < SPECTRUM_SAMPLES; i++)
		if (albedo[i] > m_mediumSamplingWeight && sig_t[i] != 0)
			m_mediumSamplingWeight = albedo[i];
	if (m_mediumSamplingWeight > 0)
		m_mediumSamplingWeight = max(m_mediumSamplingWeight, 0.5f);
	float sampledDistance = FLT_MAX;
	unsigned int channel;
	MonteCarlo::sampleReuse(SPECTRUM_SAMPLES, rand, channel);
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
		if (mRec.p == ray.ori())
			success = false;
	}
	else
	{
		sampledDistance = maxT - minT;
		success = false;
	}

	Spectrum t = (-sig_t * sampledDistance).exp();
	mRec.pdfFailure = t.average();
	mRec.pdfSuccess = (sig_t * t).average();
	mRec.transmittance = (sig_t * (-sampledDistance)).exp();
	mRec.pdfSuccessRev = mRec.pdfSuccess = mRec.pdfSuccess * m_mediumSamplingWeight;
	mRec.pdfFailure = m_mediumSamplingWeight * mRec.pdfFailure + (1 - m_mediumSamplingWeight);
	if (mRec.transmittance.max() < 1e-10f)
		mRec.transmittance = Spectrum(0.0f);

	return success;
}

DenseVolGridBaseType::DenseVolGridBaseType(Stream<char>* a_Buffer, Vec3u dim, size_t sizePerElement, size_t alignment)
{
	StreamReference<char> streamRef = a_Buffer->malloc_aligned(dim.x * dim.y * dim.z * (unsigned int)sizePerElement, (unsigned int)alignment);
	data = streamRef.AsVar<char>();
}

void DenseVolGridBaseType::InvalidateDeviceData(Stream<char>* a_Buffer)
{
	a_Buffer->translate(data).Invalidate();
}

VolumeGrid::VolumeGrid()
	: BaseVolumeRegion(CreateAggregate<PhaseFunction>(IsotropicPhaseFunction()), float4x4::Identity()), sigAMin(0.0f), sigSMin(0.0f), leMin(0.0f), sigAMax(0.0f), sigSMax(0.0f), leMax(0.0f),
	grid(), singleGrid(true)
{
	VolumeGrid::Update();
}

VolumeGrid::VolumeGrid(const PhaseFunction& func, const float4x4& ToWorld, Stream<char>* a_Buffer, Vec3u dim)
	: BaseVolumeRegion(func, ToWorld), sigAMin(0.0f), sigSMin(0.0f), leMin(0.0f), sigAMax(0.0f), sigSMax(0.0f), leMax(0.0f), 
	  grid(a_Buffer, dim), singleGrid(true)
{
	VolumeGrid::Update();
}

VolumeGrid::VolumeGrid(const PhaseFunction& func, const float4x4& ToWorld, Stream<char>* a_Buffer, Vec3u dimA, Vec3u dimS, Vec3u dimL)
	: BaseVolumeRegion(func, ToWorld), sigAMin(0.0f), sigSMin(0.0f), leMin(0.0f), sigAMax(0.0f), sigSMax(0.0f), leMax(0.0f), 
	  gridA(a_Buffer, dimA), gridS(a_Buffer, dimS), gridL(a_Buffer, dimL), singleGrid(false)
{
	VolumeGrid::Update();
}

void VolumeGrid::Update()
{
	BaseVolumeRegion::Update();
	float dimf[] = { (float)grid.dim.x - 1, (float)grid.dim.y - 1, (float)grid.dim.z - 1 };
	if (!singleGrid)
	{
		uint3 dims[] = {gridA.dim, gridS.dim, gridL.dim};
		dimf[0] = dimf[1] = dimf[2] = 0;
		for (int i = 0; i < 3; i++)
		{
			dimf[0] = max(dimf[0], float(dims[i].x - 1));
			dimf[1] = max(dimf[1], float(dims[i].y - 1));
			dimf[2] = max(dimf[2], float(dims[i].z - 1));
		}
	}
	m_stepSize = FLT_MAX;
	Vec3f size = VolumeToWorld.Scale();
	for (int i = 0; i < 3; i++)
		m_stepSize = min(m_stepSize, size[i] / dimf[i]);
	m_stepSize /= 2.0f;
}

Spectrum VolumeGrid::tau(const Ray &ray, const float minT, const float maxT) const
{
	float t0, t1;
	float length = CudaTracerLib::length(ray.dir());
	if (length == 0.f) return 0.f;
	Ray rn(ray.ori(), ray.dir() / length);
	if (!IntersectP(rn, minT * length, maxT * length, &t0, &t1)) return 0.0f;
	return integrateDensity(rn, t0, t1);
}

Spectrum VolumeGrid::integrateDensity(const Ray& ray, float t0, float t1) const
{
	Ray rayL = ray * WorldToVolume;
	float Td = rayL.dir().length();
	float minTL = t0 * Td, maxTL = t1 * Td;
	rayL.dir() = normalize(rayL.dir());
	float D_s = 0.0f, D_a = 0.0f;
	Vec3f cell_size = Vec3f(1) / grid.dimF, dir = rayL.dir() / cell_size;
	TraverseGrid(rayL, minTL, maxTL, [&](float minT, float rayT, float maxT, float cellEndT, Vec3u& cell_pos, bool& cancelTraversal)
	{
		float d_s, d_a;
		if (singleGrid)
			d_s = d_a = grid.sampleTrilinear(grid.dimF * rayL(rayT)) + grid.sampleTrilinear(grid.dimF * rayL(cellEndT));
		else
		{
			d_s = gridS.sampleTrilinear(gridS.dimF * rayL(rayT)) + gridS.sampleTrilinear(gridS.dimF * rayL(cellEndT));
			d_a = gridA.sampleTrilinear(gridA.dimF * rayL(rayT)) + gridA.sampleTrilinear(gridA.dimF * rayL(cellEndT));
		}
		d_s /= 2; d_a /= 2;
		D_s += d_s * (cellEndT - rayT);
		D_a += d_a * (cellEndT - rayT);
	}, AABB(Vec3f(0), Vec3f(1)), grid.dimF);
	float Lcl_To_World = (t1 - t0) / (maxTL - minTL);
	D_a *= Lcl_To_World;
	D_s *= Lcl_To_World;
	return sigAMin + (sigAMax - sigAMin) * D_s + sigSMin + (sigSMax - sigSMin) * D_a;
}

bool VolumeGrid::invertDensityIntegral(const Ray& ray, float t0, float t1, float desiredDensity,
									   float& integratedDensity, float &t, float &densityAtMinT, float &densityAtT) const
{
	integratedDensity = densityAtMinT = densityAtT = 0.0f;
	Ray rayL = ray * WorldToVolume;
	float Td = rayL.dir().length();
	float minTL = t0 * Td, maxTL = t1 * Td;
	rayL.dir() = normalize(rayL.dir());
	bool found = false;
	densityAtMinT = sigma_t(ray(t0), NormalizedT<Vec3f>(rayL.dir())).average();
	float Lcl_To_World = (t1 - t0) / (maxTL - minTL);
	TraverseGrid(rayL, minTL, maxTL, [&](float minT, float rayT, float maxT, float cellEndT, Vec3u& cell_pos, bool& cancelTraversal)
	{
		float d_s, d_a;
		if (singleGrid)
			d_s = d_a = grid.sampleTrilinear(grid.dimF * rayL(rayT)) + grid.sampleTrilinear(grid.dimF * rayL(cellEndT));
		else
		{
			d_s = gridS.sampleTrilinear(gridS.dimF * rayL(rayT)) + gridS.sampleTrilinear(gridS.dimF * rayL(cellEndT));
			d_a = gridA.sampleTrilinear(gridA.dimF * rayL(rayT)) + gridA.sampleTrilinear(gridA.dimF * rayL(cellEndT));
		}
		d_s /= 2; d_a /= 2;
		d_s = Spectrum(sigSMin + (sigSMax - sigSMin) * d_s).average();
		d_a = Spectrum(sigAMin + (sigAMax - sigAMin) * d_s).average();

		float D = (d_s + d_a) * (cellEndT - rayT) * Lcl_To_World;
		if (integratedDensity + D >= desiredDensity)
		{
			densityAtT = d_s + d_a;
			t = (desiredDensity - integratedDensity) / densityAtT + rayT * Lcl_To_World;
			integratedDensity = desiredDensity;
			found = true;
			cancelTraversal = true;
		}
		else
		{
			integratedDensity += D;
		}
	}, AABB(Vec3f(0), Vec3f(1)), grid.dimF);
	return found;
}

bool VolumeGrid::sampleDistance(const Ray& ray, float minT, float maxT, float sample, MediumSamplingRecord& mRec) const
{
	float t0, t1;
	float length = CudaTracerLib::length(ray.dir());
	if (length == 0.f) return 0.f;
	Ray rn(ray.ori(), ray.dir() / length);
	if (!IntersectP(rn, minT * length, maxT * length, &t0, &t1)) return false;
	float integratedDensity, densityAtMinT, densityAtT;
	float desiredDensity = -logf(1 - sample);
	bool success = false;
	if (invertDensityIntegral(rn, t0, t1, desiredDensity, integratedDensity, mRec.t, densityAtMinT, densityAtT))
	{
		success = true;
		mRec.p = ray(mRec.t);
		mRec.sigmaS = sigma_s(mRec.p, NormalizedT<Vec3f>(-ray.dir()));
		mRec.sigmaA = sigma_s(mRec.p, NormalizedT<Vec3f>(-ray.dir()));
	}
	float expVal = math::exp(-integratedDensity);
	mRec.pdfFailure = expVal;
	mRec.pdfSuccess = expVal * densityAtT;
	mRec.pdfSuccessRev = expVal * densityAtMinT;
	mRec.transmittance = Spectrum(expVal);
	return success && mRec.pdfSuccess > 0;
}

bool KernelAggregateVolume::IntersectP(const Ray &ray, float minT, float maxT, float *t0, float *t1) const
{
	*t0 = FLT_MAX;
	*t1 = -FLT_MAX;
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
	{
		float a, b;
		if (m_pVolumes[i].IntersectP(ray, minT, maxT, &a, &b))
		{
			*t0 = min(*t0, a);
			*t1 = max(*t1, b);
		}
	}
	return (*t0 < *t1);
}

Spectrum KernelAggregateVolume::sigma_a(const Vec3f& p, const NormalizedT<Vec3f>& w) const
{
	Spectrum s = Spectrum(0.0f);
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
			s += m_pVolumes[i].sigma_a(p, w);
	return s;
}

Spectrum KernelAggregateVolume::sigma_s(const Vec3f& p, const NormalizedT<Vec3f>& w) const
{
	Spectrum s = Spectrum(0.0f);
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
			s += m_pVolumes[i].sigma_s(p, w);
	return s;
}

Spectrum KernelAggregateVolume::Lve(const Vec3f& p, const NormalizedT<Vec3f>& w) const
{
	Spectrum s = Spectrum(0.0f);
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
			s += m_pVolumes[i].Lve(p, w);
	return s;
}

Spectrum KernelAggregateVolume::sigma_t(const Vec3f &p, const NormalizedT<Vec3f> &wo) const
{
	Spectrum s = Spectrum(0.0f);
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
			s += m_pVolumes[i].sigma_t(p, wo);
	return s;
}

Spectrum KernelAggregateVolume::tau(const Ray &ray, float minT, float maxT) const
{
	Spectrum s = Spectrum(0.0f);
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
			s += m_pVolumes[i].tau(ray, minT, maxT);
	return s;
}

float KernelAggregateVolume::Sample(const Vec3f& p, const NormalizedT<Vec3f>& wo, CudaRNG& rng, NormalizedT<Vec3f>* wi) const
{
	float vol_sample_pdf = 0; float sample = rng.randomFloat();
	const auto* vol = sampleVolume(Ray(p, wo), 0, FLT_MAX, sample, vol_sample_pdf);
	if (vol)
	{
		float sample_pdf;
		PhaseFunctionSamplingRecord pRec(wo, *wi);
		float pf_val = vol->As()->Func.Sample(pRec, sample_pdf, rng.randomFloat2());
		*wi = pRec.wo;
		return pf_val;
	}
	else return 0.0f;
}

float KernelAggregateVolume::p(const Vec3f& p, const NormalizedT<Vec3f>& wo, const NormalizedT<Vec3f>& wi, CudaRNG& rng) const
{
	PhaseFunctionSamplingRecord r2(wi, wo);
	float ph = 0, sumWt = 0;
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
		if (m_pVolumes[i].WorldBound().Contains(p))
		{
			float wt = m_pVolumes[i].sigma_s(p, wo).average();
			sumWt += wt;
			ph += wt * m_pVolumes[i].As()->Func.Evaluate(r2);
		}
	return sumWt != 0 ? ph / sumWt : 0.0f;
}

//http://stackoverflow.com/questions/7669057/find-nth-set-bit-in-an-int
CUDA_FUNC_IN int ffsn(unsigned int v, int n) {
	for (int i = 0; i<n - 1; i++) {
		v &= v - 1; // remove the least significant bit
	}
	return v & ~(v - 1); // extract the least significant bit
}

bool KernelAggregateVolume::sampleDistance(const Ray& ray, float minT, float maxT, CudaRNG& rng, MediumSamplingRecord& mRec) const
{
	float vol_sample_pdf = 0; float sample = rng.randomFloat();
	const auto* vol = sampleVolume(ray, minT, maxT, sample, vol_sample_pdf);
	if(vol && vol->sampleDistance(ray, minT, maxT, rng.randomFloat(), mRec))
	{
		//mRec.pdfSuccess *= vol_sample_pdf;
		//mRec.pdfSuccessRev *= vol_sample_pdf;
		//mRec.pdfFailure *= vol_sample_pdf;
		return true;
	}
	else return false;
}

KernelAggregateVolume::KernelAggregateVolume(Stream<VolumeRegion>* D, bool devicePointer)
{
	m_uVolumeCount = 0;
	for (Stream<VolumeRegion>::iterator it = D->begin(); it != D->end(); ++it)
	{
		m_pVolumes[m_uVolumeCount] = *(*it);
		m_uVolumeCount++;
	}
	box = AABB::Identity();
	for (unsigned int i = 0; i < m_uVolumeCount; i++)
		box = box.Extend(D->operator()(i)->WorldBound());
}

const VolumeRegion* KernelAggregateVolume::sampleVolume(const Ray& ray, float minT, float maxT, float& sample, float& pdf) const
{
	if (m_uVolumeCount == 0)
		return 0;
	else if (m_uVolumeCount == 1)
		return m_pVolumes[0].WorldBound().Intersect<true>(ray, &minT, &maxT) ? m_pVolumes : 0;

	unsigned int n = 0;
	unsigned int flag = 0;
	for (unsigned int i = 0; i < m_uVolumeCount; i++)
		if (m_pVolumes[i].WorldBound().Intersect<true>(ray, &minT, &maxT))//the pointers will only be set in case true is returned
		{
			n++;
			flag |= 1 << i;
		}
	if (!n)
		return 0;
	unsigned int nth;
	MonteCarlo::sampleReuse(n, sample, nth);
	int i = ffsn(flag, nth);
	pdf = 1.0f / n;
	return m_pVolumes + i;
}

}
