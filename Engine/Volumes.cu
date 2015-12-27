#include "Buffer.h"
#include "Volumes.h"
#include <Base/CudaRandom.h>
#include "Samples.h"
#include "Grid.h"

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
	for (int i = 0; i < 2; i++)
		if (albedo[i] > m_mediumSamplingWeight && sig_t[i] != 0)
			m_mediumSamplingWeight = albedo[i];
	if (m_mediumSamplingWeight > 0)
		m_mediumSamplingWeight = max(m_mediumSamplingWeight, 0.5f);
	float sampledDistance = FLT_MAX;
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

VolumeGrid::VolumeGrid(const PhaseFunction& func, const float4x4& ToWorld, Stream<char>* a_Buffer, Vec3u dim)
	: grid(a_Buffer, dim), singleGrid(true)
{
	VolumeToWorld = ToWorld;
	BaseVolumeRegion::Func = func;
	sigAMin = sigSMin = leMin = Spectrum(0.0f);
	sigAMax = sigSMax = leMax = Spectrum(1.0f);
	Update();
}

VolumeGrid::VolumeGrid(const PhaseFunction& func, const float4x4& ToWorld, Stream<char>* a_Buffer, Vec3u dimA, Vec3u dimS, Vec3u dimL)
	: gridA(a_Buffer, dimA), gridS(a_Buffer, dimS), gridL(a_Buffer, dimL), singleGrid(false)
{
	VolumeToWorld = ToWorld;
	BaseVolumeRegion::Func = func;
	sigAMin = sigSMin = leMin = Spectrum(0.0f);
	sigAMax = sigSMax = leMax = Spectrum(1.0f);
	Update();
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
	float length = CudaTracerLib::length(ray.direction);
	if (length == 0.f) return 0.f;
	Ray rn(ray.origin, ray.direction / length);
	if (!IntersectP(rn, minT * length, maxT * length, &t0, &t1)) return 0.;
	float f = integrateDensity(rn, t0, t1);
	return sigAMin + (sigAMax - sigAMin) * f + sigSMin + (sigSMax - sigSMin) * f;
	/*
	Spectrum tau(0.);
	t0 += u * stepSize;
	while (t0 < t1) {
		tau += sigma_t(rn(t0), -rn.direction);
		t0 += stepSize;
	}
	return tau * stepSize;*/
	/*float t0, t1;
	if (!IntersectP(ray, minT, maxT, &t0, &t1))
		return Spectrum(0.0f);
	Spectrum tau(0.0f);
	int N = 10;
	float step = (t1 - t0) / float(N);
	t0 += step / 2;
	int i = -1;
	while (t0 < t1 && i++ < N)
	{
		tau += sigma_t(ray(t0), -ray.direction);
		t0 += step;
	}
	return tau * step;*/
}

float VolumeGrid::integrateDensity(const Ray& ray, float t0, float t1) const
{
#if 0
	float length = t1 - t0, maxComp = 0;
	Vec3f p = ray(t0), pLast = ray(t1);
	float pf[] = { p.x, p.y, p.z };
	float pLastf[] = { pLast.x, pLast.y, pLast.z };
	for (int i = 0; i<3; ++i)
		maxComp = max(max(maxComp, math::abs(pf[i])), math::abs(pLastf[i]));
	if (length < 1e-6f * maxComp)
		return 0.0f;
	float m_scale = 1.0f;
	unsigned int nSteps = (unsigned int)ceilf(length / (2 * m_stepSize));
	nSteps += nSteps % 2;
	float stepSize = length / nSteps;
	const Vec3f increment = ray.direction * stepSize;
	float integratedDensity = densityT(p) + densityT(pLast);
	p += increment;
	float m = 4;
	for (unsigned int i = 1; i < nSteps; ++i)
	{
		integratedDensity += m * densityT(p);
		m = 6 - m;
		Vec3f next = p + increment;
		if (p == next)
		{
			printf("integrateDensity() not stepping forward, stepsize = %f.\n", stepSize);
			break;
		}
		p = next;
	}
	return integratedDensity * m_scale * stepSize * (1.0f / 3.0f);
#endif

	Ray rayL = ray * WorldToVolume;
	float Td = rayL.direction.length();
	float minTL = t0 * Td, maxTL = t1 * Td;
	rayL.direction.normalize();
	float integratedDensity = 0;
	TraverseGrid(rayL, minTL, maxTL, [&](float minT, float rayT, float maxT, float cellEndT, Vec3u& cell_pos, bool& cancelTraversal)
	{
		float d0 = 2 * grid.sampleTrilinear(grid.dimF * rayL(rayT)), d1 = 2 * grid.sampleTrilinear(grid.dimF * rayL(cellEndT)), d = (d0 + d1) / 2.0f;
		integratedDensity += d * (cellEndT - rayT);
		/*
		//Performs analytic integration over the cell, somewhat buggy
		unsigned char xFlag = 0x66, zFlag = 0xcc;
		float V[8];
		for (int i = 0; i < 8; i++)
			V[i] = grid.value(cell_pos.x + ((xFlag >> i) & 1), cell_pos.y + i / 4, cell_pos.z + ((zFlag >> i) & 1));
		Vec3f r = rayL(rayT) * grid.dimF - Vec3f(cell_pos.x, cell_pos.y, cell_pos.z);
		float t1 = cellEndT - rayT, t2 = t1 * t1, t3 = t1 * t1 * t1;
		Vec3f R = r * t1 + 0.5f * rayL.direction * t2;
		float Rxy = r.x * r.y * t1 + 0.5f * r.x * rayL.direction.y * t2 + 0.5f * r.y * rayL.direction.x * t2 + 1.0f / 3.0f * rayL.direction.x * rayL.direction.y * t3;
		float Ryz = r.y * r.z * t1 + 0.5f * r.y * rayL.direction.z * t2 + 0.5f * r.z * rayL.direction.y * t2 + 1.0f / 3.0f * rayL.direction.y * rayL.direction.z * t3;
		float Rxz = r.x * r.z * t1 + 0.5f * r.x * rayL.direction.z * t2 + 0.5f * r.z * rayL.direction.x * t2 + 1.0f / 3.0f * rayL.direction.x * rayL.direction.z * t3;
		integratedDensity += t1 * V[0] + (V[1] - V[0]) * R.x + (V[3] - V[0]) * R.z + (V[2] + V[0] - V[3] - V[1]) * Rxz;
		integratedDensity += (V[0] - V[4]) * R.y + (V[1] - V[0] - V[5] + V[4]) * Rxy + (V[3] - V[0] - V[7] + V[4]) * Ryz;*/
	}, AABB(Vec3f(0), Vec3f(1)), grid.dimF);
	return integratedDensity * (t1 - t0) / (maxTL - minTL);

}

bool VolumeGrid::invertDensityIntegral(const Ray& ray, float t0, float t1, float desiredDensity,
										 float &integratedDensity, float &t, float &densityAtMinT, float &densityAtT) const
{

	integratedDensity = densityAtMinT = densityAtT = 0.0f;
	Ray rayL = ray * WorldToVolume;
	float Td = rayL.direction.length();
	float minTL = t0 * Td, maxTL = t1 * Td;
	rayL.direction.normalize();
	bool found = false;
	densityAtMinT = sigma_t(ray(t0), Vec3f(0)).average();
	TraverseGrid(rayL, minTL, maxTL, [&](float minT, float rayT, float maxT, float cellEndT, Vec3u& cell_pos, bool& cancelTraversal)
	{
		float d0 = grid.sampleTrilinear(grid.dimF * rayL(rayT)) * (sigAMax + sigSMax).average(),
			  d1 = grid.sampleTrilinear(grid.dimF * rayL(cellEndT)) * (sigAMax + sigSMax).average(),
			  d = (d0 + d1) / 2.0f;
		float D = d * (cellEndT - rayT) * (t1 - t0) / (maxTL - minTL);
		if(integratedDensity + D > desiredDensity)
		{
			densityAtT = d;
			t = (desiredDensity - integratedDensity) / d + rayT * (t1 - t0) / (maxTL - minTL);
			integratedDensity = desiredDensity;
			found = true;
			cancelTraversal = true;
		}
		else integratedDensity += D;
	}, AABB(Vec3f(0), Vec3f(1)), grid.dimF);
	return found;

#if 0
	integratedDensity = densityAtMinT = densityAtT = 0.0f;
	float length = t1 - t0, maxComp = 0;
	Vec3f p = ray(t0), pLast = ray(t1);
	for (int i = 0; i<3; ++i)
		maxComp = max(max(maxComp, math::abs(p[i])), math::abs(pLast[i]));
	if (length < 1e-6f * maxComp)
		return 0.0f;
	float m_scale = 1;
	unsigned int nSteps = (unsigned int)ceilf(length / (2 * m_stepSize));
	float stepSize = length / nSteps, multiplier = (1.0f / 6.0f) * stepSize * m_scale;
	Vec3f fullStep = ray.direction * stepSize, halfStep = fullStep * .5f;
	float node1 = sigma_t(p, Vec3f(0)).average();
	densityAtMinT = node1 * m_scale;
	for (unsigned int i = 0; i < nSteps; ++i)
	{
		float node2 = sigma_t(p + halfStep, Vec3f(0.0f)).average(),
			  node3 = sigma_t(p + fullStep, Vec3f(0.0f)).average();
		float newDensity = integratedDensity + multiplier * (node1 + node2 * 4 + node3);
		if (newDensity >= desiredDensity)
		{
			/*float a = 0, b = stepSize, x = a,
				fx = integratedDensity - desiredDensity,
				stepSizeSqr = stepSize * stepSize,
				temp = m_scale / stepSizeSqr;
			int it = 1;
			while (true)
			{
				float dfx = temp * (node1 * stepSizeSqr
					- (3 * node1 - 4 * node2 + node3)*stepSize*x
					+ 2 * (node1 - 2 * node2 + node3)*x*x);
				x -= fx / dfx;
				if (x <= a || x >= b || dfx == 0)
					x = 0.5f * (b + a);
				float intval = integratedDensity + temp * (1.0f / 6.0f) * (x *
					(6 * node1*stepSizeSqr - 3 * (3 * node1 - 4 * node2 + node3)*stepSize*x
					+ 4 * (node1 - 2 * node2 + node3)*x*x));
				fx = intval - desiredDensity;

				if (math::abs(fx) < 1e-3f) {
					t = minT + stepSize * i + x;
					integratedDensity = intval;
					densityAtT = temp * (node1 * stepSizeSqr
						- (3 * node1 - 4 * node2 + node3)*stepSize*x
						+ 2 * (node1 - 2 * node2 + node3)*x*x);
					return true;
				}
				else if (++it > 30)
				{
					printf("invertDensityIntegral(): stuck in Newton-Bisection -- "
							"round-off error issues? The step size was %e, fx=%f, dfx=%f, "
							"a=%f, b=%f", stepSize, fx, dfx, a, b);
					return false;
				}
				if (fx > 0)
					b = x;
				else
					a = x;
			}*/
			float V = desiredDensity - integratedDensity, s = (node3 - node1) / (2 * stepSize);
			float r = (node1 * node1) / (4 * stepSize * stepSize) + V / stepSize;
			if (r < 0)
				printf("r = %f\n", r);
			float tl = -node1 / (2 * stepSize) + math::sqrt(r);
			t = t0 + stepSize * i + tl;
			integratedDensity = desiredDensity;
			densityAtT = s * tl + node1;
			return true;
		}
		Vec3f next = p + fullStep;
		if (p == next)
		{
			printf("invertDensityIntegral() not stepping forward, stepsize = %f.\n", stepSize);
			break;
		}
		integratedDensity = newDensity;
		node1 = node3;
		p = next;
	}
	return false;
#endif
}

bool VolumeGrid::sampleDistance(const Ray& ray, float minT, float maxT, float sample, MediumSamplingRecord& mRec) const
{
	float t0, t1;
	float length = CudaTracerLib::length(ray.direction);
	if (length == 0.f) return 0.f;
	Ray rn(ray.origin, ray.direction / length);
	if (!IntersectP(rn, minT * length, maxT * length, &t0, &t1)) return 0.;
	float integratedDensity, densityAtMinT, densityAtT;
	float desiredDensity = -logf(1 - sample);
	bool success = false;
	if (invertDensityIntegral(rn, t0, t1, desiredDensity, integratedDensity, mRec.t, densityAtMinT, densityAtT))
	{
		success = true;
		mRec.p = ray(mRec.t);
		mRec.sigmaS = sigma_s(mRec.p, -ray.direction);
		mRec.sigmaA = sigma_s(mRec.p, -ray.direction);
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

Spectrum KernelAggregateVolume::sigma_a(const Vec3f& p, const Vec3f& w) const
{
	Spectrum s = Spectrum(0.0f);
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
			s += m_pVolumes[i].sigma_a(p, w);
	return s;
}

Spectrum KernelAggregateVolume::sigma_s(const Vec3f& p, const Vec3f& w) const
{
	Spectrum s = Spectrum(0.0f);
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
			s += m_pVolumes[i].sigma_s(p, w);
	return s;
}

Spectrum KernelAggregateVolume::Lve(const Vec3f& p, const Vec3f& w) const
{
	Spectrum s = Spectrum(0.0f);
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
			s += m_pVolumes[i].Lve(p, w);
	return s;
}

Spectrum KernelAggregateVolume::sigma_t(const Vec3f &p, const Vec3f &wo) const
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

float KernelAggregateVolume::Sample(const Vec3f& p, const Vec3f& wo, CudaRNG& rng, Vec3f* wi)
{
	PhaseFunctionSamplingRecord r2(wo);
	r2.wi = wo;
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
		if (m_pVolumes[i].WorldBound().Contains(p))
		{
			float pdf;
			float pf = m_pVolumes[i].As()->Func.Sample(r2, pdf, rng);
			*wi = r2.wo;
			return pf;
		}
		
	return 0.0f;
}

float KernelAggregateVolume::p(const Vec3f& p, const Vec3f& wo, const Vec3f& wi, CudaRNG& rng)
{
	PhaseFunctionSamplingRecord r2(wo, wi);
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
		if (m_pVolumes[i].WorldBound().Contains(p))
			return m_pVolumes[i].As()->Func.Evaluate(r2);
	return 0.0f;
}

CUDA_FUNC_IN int ffsn(unsigned int v, int n) {
	for (int i = 0; i<n - 1; i++) {
		v &= v - 1; // remove the least significant bit
	}
	return v & ~(v - 1); // extract the least significant bit
}

bool KernelAggregateVolume::sampleDistance(const Ray& ray, float minT, float maxT, CudaRNG& rng, MediumSamplingRecord& mRec) const
{
	if (m_uVolumeCount == 1 && m_pVolumes[0].WorldBound().Intersect(ray))
		return m_pVolumes[0].sampleDistance(ray, minT, maxT, rng.randomFloat(), mRec);
	else if (m_uVolumeCount == 1)
		return false;

	float n = 0;
	unsigned int flag = 0;
	for (unsigned int i = 0; i < m_uVolumeCount; i++)
		if (m_pVolumes[i].WorldBound().Intersect(ray))
		{
			n++;
			flag |= 1 << i;
		}
	if (!n)
		return 0;
	float sample = rng.randomFloat();
	int nth = int(sample * n);
	int i = ffsn(flag, nth);
	return m_pVolumes[i].sampleDistance(ray, minT, maxT, rng.randomFloat(), mRec);
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

}
