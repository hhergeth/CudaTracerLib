#include "e_Volumes.h"

bool e_HomogeneousVolumeDensity::IntersectP(const Ray &ray, const float minT, const float maxT, float *t0, float *t1) const
{
	Ray r = ray * WorldToVolume;
	bool b = e_BaseVolumeRegion::Box.Intersect(r, t0, t1);
	if(b)
	{
		*t0 = clamp(*t0, minT, maxT);
		*t1 = clamp(*t1, minT, maxT);
	}
	return b && *t1 > *t0 && *t1 > 0;
}

Spectrum e_HomogeneousVolumeDensity::tau(const Ray &ray, const float minT, const float maxT) const
{
	float t0, t1;
	if(!IntersectP(ray, minT, maxT, &t0, &t1))
		return Spectrum(0.0f);
	return length(ray(t0) - ray(t1)) * (sig_a + sig_s);
}

bool e_HomogeneousVolumeDensity::sampleDistance(const Ray& ray, float minT, float maxT, float rand, MediumSamplingRecord& mRec) const
{
	return sampleDistanceHomogenous(ray, minT, maxT, rand, mRec, sig_a, sig_s);
}

e_VolumeGrid::e_VolumeGrid(const e_PhaseFunction& func, const float4x4 volToWorld, e_Stream<char>* a_Buffer, uint3 dim)
	: grid(a_Buffer, dim), singleGrid(true)
{
	VolumeToWorld = volToWorld;
	WorldToVolume = volToWorld.inverse();
	e_BaseVolumeRegion::Func = func;
	sigAMin = sigSMin = leMin = Spectrum(0.0f);
	sigAMax = sigSMax = leMax = Spectrum(1.0f);
	Update();
}

e_VolumeGrid::e_VolumeGrid(const e_PhaseFunction& func, const float4x4 volToWorld, e_Stream<char>* a_Buffer, uint3 dimA, uint3 dimS, uint3 dimL)
	: gridA(a_Buffer, dimA), gridS(a_Buffer, dimS), gridL(a_Buffer, dimL), singleGrid(false)
{
	VolumeToWorld = volToWorld;
	WorldToVolume = volToWorld.inverse();
	e_BaseVolumeRegion::Func = func;
	sigAMin = sigSMin = leMin = Spectrum(0.0f);
	sigAMax = sigSMax = leMax = Spectrum(1.0f);
	Update();
}

void e_VolumeGrid::Update()
{
	e_BaseVolumeRegion::Box = AABB(make_float3(0), make_float3(1)).Transform(VolumeToWorld);
	float dimf[] = { grid.dim.x - 1, grid.dim.y - 1, grid.dim.z - 1 };
	if (!singleGrid)
	{
		uint3 dims[] = {gridA.dim, gridS.dim, gridL.dim};
		dimf[0] = dimf[1] = dimf[2] = 0;
		for (int i = 0; i < 3; i++)
		{
			dimf[0] = MAX(dimf[0], float(dims[i].x - 1));
			dimf[1] = MAX(dimf[1], float(dims[i].y - 1));
			dimf[2] = MAX(dimf[2], float(dims[i].z - 1));
		}
	}
	m_stepSize = FLT_MAX;
	for (int i = 0; i < 3; i++)
		m_stepSize = MIN(m_stepSize, (Box.max[i] - Box.min[i]) / dimf[i]);
	m_stepSize /= 2.0f;
}

bool e_VolumeGrid::IntersectP(const Ray &ray, const float minT, const float maxT, float *t0, float *t1) const
{
	bool b = e_BaseVolumeRegion::Box.Intersect(ray, t0, t1);
	if (b)
	{
		*t0 = clamp(*t0, minT, maxT);
		*t1 = clamp(*t1, minT, maxT);
	}
	return b && *t1 > *t0 && *t1 > 0;
}

Spectrum e_VolumeGrid::tau(const Ray &ray, const float minT, const float maxT) const
{
	float t0, t1;
	float length = ::length(ray.direction);
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

float e_VolumeGrid::integrateDensity(const Ray& ray, float minT, float maxT) const
{
	float length = maxT - minT, maxComp = 0;
	float3 p = ray(minT), pLast = ray(maxT);
	float pf[] = { p.x, p.y, p.z };
	float pLastf[] = { pLast.x, pLast.y, pLast.z };
	for (int i = 0; i<3; ++i)
		maxComp = MAX(MAX(maxComp, fabsf(pf[i])), fabsf(pLastf[i]));
	if (length < 1e-6f * maxComp)
		return 0.0f;
	float m_scale = 1.0f;
	unsigned int nSteps = (unsigned int)ceilf(length / (2 * m_stepSize));
	nSteps += nSteps % 2;
	float stepSize = length / nSteps;
	const float3 increment = ray.direction * stepSize;
	float integratedDensity = densityT(p) + densityT(pLast);
	p += increment;
	float m = 4;
	for (unsigned int i = 1; i < nSteps; ++i)
	{
		integratedDensity += m * densityT(p);
		m = 6 - m;
		float3 next = p + increment;
		if (p == next)
		{
			printf("integrateDensity() not stepping forward, stepsize = %f.\n", stepSize);
			break;
		}
		p = next;
	}
	return integratedDensity * m_scale * stepSize * (1.0f / 3.0f);
}

bool e_VolumeGrid::invertDensityIntegral(const Ray& ray, float minT, float maxT, float desiredDensity,
										 float &integratedDensity, float &t, float &densityAtMinT, float &densityAtT) const
{
	integratedDensity = densityAtMinT = densityAtT = 0.0f;
	float length = maxT - minT, maxComp = 0;
	float3 p = ray(minT), pLast = ray(maxT);
	float pf[] = {p.x, p.y, p.z};
	float pLastf[] = { pLast.x, pLast.y, pLast.z };
	for (int i = 0; i<3; ++i)
		maxComp = MAX(MAX(maxComp, fabsf(pf[i])), fabsf(pLastf[i]));
	if (length < 1e-6f * maxComp)
		return 0.0f;
	float m_scale = 1;
	unsigned int nSteps = (unsigned int)ceilf(length / (2 * m_stepSize));
	float stepSize = length / nSteps, multiplier = (1.0f / 6.0f) * stepSize * m_scale;
	float3 fullStep = ray.direction * stepSize,	halfStep = fullStep * .5f;
	float node1 = densityT(p);
	densityAtMinT = node1 * m_scale;
	for (unsigned int i = 0; i < nSteps; ++i)
	{
		float node2 = densityT(p + halfStep),
			  node3 = densityT(p + fullStep),
			  newDensity = integratedDensity + multiplier * (node1 + node2 * 4 + node3);
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

				if (fabsf(fx) < 1e-3f) {
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
			//float c = 0, a = (node3 - node1 - 2 * (node2 - node1)) / (stepSize * stepSize * 0.5f), b = (node3 - node1 - a * stepSize * stepSize) / stepSize, d = desiredDensity - integratedDensity - node1;
			//float q0 = (1.73205080757f*sqrtf(d*(6 * a * a * d - b * b * b)) / (2.82842712475f * a * a) + (12 * a * a * d - b * b * b) / (8 * a * a * a)), q1 = powf(q0, 1.0f / 3.0f);
			//t = minT + stepSize * i + q1 + b * b / (4 * a * a * q1) - b / (2 * a);
			//return true;
			float V = desiredDensity - integratedDensity, s = (node3 - node1) / (2 * stepSize);
			float r = (node1 * node1) / (4 * stepSize * stepSize) + V / stepSize;
			if (r < 0)
				printf("r = %f\n", r);
			float tl = -node1 / (2 * stepSize) + sqrtf(r);
			t = minT + stepSize * i + tl;
			integratedDensity = desiredDensity;
			densityAtT = s * tl + node1;
			return true;
		}
		float3 next = p + fullStep;
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
}

bool e_VolumeGrid::sampleDistance(const Ray& ray, float minT, float maxT, float sample, MediumSamplingRecord& mRec) const
{
	float t0, t1;
	float length = ::length(ray.direction);
	if (length == 0.f) return 0.f;
	Ray rn(ray.origin, ray.direction / length);
	if (!IntersectP(rn, minT * length, maxT * length, &t0, &t1)) return 0.;
	float integratedDensity, densityAtMinT, densityAtT;
	float desiredDensity = -logf(1 - sample);
	float sigta = sigma_t(rn(t0), rn.direction).average();
	bool success = false;
	if (invertDensityIntegral(rn, t0, t1, desiredDensity, integratedDensity, mRec.t, densityAtMinT, densityAtT))
	{
		success = true;
		mRec.p = ray(mRec.t);
		mRec.sigmaS = sigma_s(mRec.p, -ray.direction);
		mRec.sigmaA = sigma_s(mRec.p, -ray.direction);
	}
	float expVal = expf(-integratedDensity);
	mRec.pdfFailure = expVal;
	mRec.pdfSuccess = expVal * densityAtT;
	mRec.pdfSuccessRev = expVal * densityAtMinT;
	mRec.transmittance = Spectrum(expVal);
	return success && mRec.pdfSuccess > 0;
}

bool e_KernelAggregateVolume::IntersectP(const Ray &ray, float minT, float maxT, unsigned int a_NodeIndex, float *t0, float *t1) const
{
	*t0 = FLT_MAX;
	*t1 = -FLT_MAX;
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
	{
		float a, b;
		if (m_pVolumes[i].As()->isInVolume(a_NodeIndex) && m_pVolumes[i].IntersectP(ray, minT, maxT, &a, &b))
		{
			*t0 = MIN(*t0, a);
			*t1 = MAX(*t1, b);
		}
	}
	return (*t0 < *t1);
}

Spectrum e_KernelAggregateVolume::sigma_a(const float3& p, const float3& w, unsigned int a_NodeIndex) const
{
	Spectrum s = Spectrum(0.0f);
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
		if (m_pVolumes[i].As()->isInVolume(a_NodeIndex))
			s += m_pVolumes[i].sigma_a(p, w);
	return s;
}

Spectrum e_KernelAggregateVolume::sigma_s(const float3& p, const float3& w, unsigned int a_NodeIndex) const
{
	Spectrum s = Spectrum(0.0f);
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
		if (m_pVolumes[i].As()->isInVolume(a_NodeIndex))
			s += m_pVolumes[i].sigma_s(p, w);
	return s;
}

Spectrum e_KernelAggregateVolume::Lve(const float3& p, const float3& w, unsigned int a_NodeIndex) const
{
	Spectrum s = Spectrum(0.0f);
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
		if (m_pVolumes[i].As()->isInVolume(a_NodeIndex))
			s += m_pVolumes[i].Lve(p, w);
	return s;
}

Spectrum e_KernelAggregateVolume::sigma_t(const float3 &p, const float3 &wo, unsigned int a_NodeIndex) const
{
	Spectrum s = Spectrum(0.0f);
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
		if (m_pVolumes[i].As()->isInVolume(a_NodeIndex))
			s += m_pVolumes[i].sigma_t(p, wo);
	return s;
}

Spectrum e_KernelAggregateVolume::tau(const Ray &ray, float minT, float maxT, unsigned int a_NodeIndex) const
{
	Spectrum s = Spectrum(0.0f);
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
		if (m_pVolumes[i].As()->isInVolume(a_NodeIndex))
			s += m_pVolumes[i].tau(ray, minT, maxT);
	return s;
}

float e_KernelAggregateVolume::Sample(const float3& p, const float3& wo, unsigned int a_NodeIndex, CudaRNG& rng, float3* wi)
{
	PhaseFunctionSamplingRecord r2(wo);
	r2.wi = wo;
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
		if (m_pVolumes[i].As()->isInVolume(a_NodeIndex) && m_pVolumes[i].WorldBound().Contains(p))
		{
			float pdf;
			float pf = m_pVolumes[i].BaseRegion()->Func.Sample(r2, pdf, rng);
			*wi = r2.wo;
			return pf;
		}
		
	return 0.0f;
}

float e_KernelAggregateVolume::p(const float3& p, const float3& wo, const float3& wi, unsigned int a_NodeIndex, CudaRNG& rng)
{
	PhaseFunctionSamplingRecord r2(wo, wi);
	r2.wi = wo;
	r2.wo = wi;
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
		if (m_pVolumes[i].As()->isInVolume(a_NodeIndex) && m_pVolumes[i].WorldBound().Contains(p))
			return m_pVolumes[i].BaseRegion()->Func.Evaluate(r2);
	return 0.0f;
}

CUDA_FUNC_IN int ffsn(unsigned int v, int n) {
	for (int i = 0; i<n - 1; i++) {
		v &= v - 1; // remove the least significant bit
	}
	return v & ~(v - 1); // extract the least significant bit
}

bool e_KernelAggregateVolume::sampleDistance(const Ray& ray, float minT, float maxT, unsigned int a_NodeIndex, CudaRNG& rng, MediumSamplingRecord& mRec) const
{
	if (m_uVolumeCount == 1 && m_pVolumes[0].As()->isInVolume(a_NodeIndex) && m_pVolumes[0].WorldBound().Intersect(ray))
		return m_pVolumes[0].sampleDistance(ray, minT, maxT, rng.randomFloat(), mRec);
	else if (m_uVolumeCount == 1)
		return false;

	float n = 0;
	unsigned int flag = 0;
	for (unsigned int i = 0; i < m_uVolumeCount; i++)
		if (m_pVolumes[i].As()->isInVolume(a_NodeIndex) && m_pVolumes[i].WorldBound().Intersect(ray))
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