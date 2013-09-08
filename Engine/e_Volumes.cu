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

Spectrum e_HomogeneousVolumeDensity::tau(const Ray &ray, const float minT, const float maxT, float step, float offset) const
{
	float t0, t1;
	if(!IntersectP(ray, minT, maxT, &t0, &t1))
		return Spectrum(0.0f);
	return length(ray(t0) - ray(t1)) * (sig_a + sig_s);
}

bool e_KernelAggregateVolume::IntersectP(const Ray &ray, const float minT, const float maxT, float *t0, float *t1) const
{
	*t0 = FLT_MAX;
	*t1 = -FLT_MAX;
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
	{
		float a, b;
		if(m_pVolumes[i].IntersectP(ray, minT, maxT, &a, &b))
		{
			*t0 = MIN(*t0, a);
			*t1 = MAX(*t1, b);
		}
	}
	return (*t0 < *t1);
}

Spectrum e_KernelAggregateVolume::sigma_a(const float3& p, const float3& w) const
{
	Spectrum s = Spectrum(0.0f);
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
		s += m_pVolumes[i].sigma_a(p, w);
	return s;
}

Spectrum e_KernelAggregateVolume::sigma_s(const float3& p, const float3& w) const
{
	Spectrum s = Spectrum(0.0f);
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
		s += m_pVolumes[i].sigma_s(p, w);
	return s;
}

Spectrum e_KernelAggregateVolume::Lve(const float3& p, const float3& w) const
{
	Spectrum s = Spectrum(0.0f);
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
		s += m_pVolumes[i].Lve(p, w);
	return s;
}

Spectrum e_KernelAggregateVolume::sigma_t(const float3 &p, const float3 &wo) const
{
	Spectrum s = Spectrum(0.0f);
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
		s += m_pVolumes[i].sigma_t(p, wo);
	return s;
}

Spectrum e_KernelAggregateVolume::tau(const Ray &ray, const float minT, const float maxT, float step, float offset) const
{
	Spectrum s = Spectrum(0.0f);
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
		s += m_pVolumes[i].tau(ray, minT, maxT, step, offset);
	return s;
}

float e_KernelAggregateVolume::Sample(const float3& p, const float3& wo, CudaRNG& rng, float3* wi)
{
	PhaseFunctionSamplingRecord r2(wo);
	r2.wi = wo;
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
		if(m_pVolumes[i].WorldBound().Contains(p))
		{
			float pdf;
			float pf = m_pVolumes[i].BaseRegion()->Func.Sample(r2, pdf, rng);
			*wi = r2.wo;
			return pdf * pf;
		}
		
	return 0.0f;
}

float e_KernelAggregateVolume::p(const float3& p, const float3& wo, const float3& wi, CudaRNG& rng)
{
	PhaseFunctionSamplingRecord r2(wo, wi);
	r2.wi = wo;
	r2.wo = wi;
	for(unsigned int i = 0; i < m_uVolumeCount; i++)
		if(m_pVolumes[i].WorldBound().Contains(p))
			return m_pVolumes[i].BaseRegion()->Func.Evaluate(r2);
	return 0.0f;
}