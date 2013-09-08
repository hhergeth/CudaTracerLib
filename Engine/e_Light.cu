#include "e_Light.h"

Spectrum e_PointLight::Sample_L(const e_KernelDynamicScene& scene, const LightSample &ls, float u1, float u2, Ray *ray, float3 *Ns, float *pdf) const
{		
	float3 p,d;
	if(radius != 0)
	{
		float3 n = UniformSampleSphere(u1, u2) * radius;
		p = lightPos + n;
		//d = SampleCosineHemisphere(n, u1, u2);
		d = UniformSampleSphere(ls.uPos[0], ls.uPos[1]);
	}
	else
	{
		p = lightPos;
		d = UniformSampleSphere(ls.uPos[0], ls.uPos[1]);
	}	
	*ray = Ray(p, d);
	*Ns = ray->direction;
	*pdf = UniformSpherePdf();
	return Intensity;
}

Spectrum e_DiffuseLight::Sample_L(const e_KernelDynamicScene& scene, const LightSample &ls, float u1, float u2, Ray *ray, float3 *Ns, float *pdf) const
{
	float3 org = shapeSet.Sample(ls, Ns, scene.m_sBVHIntData.Data);
	float3 dir = UniformSampleSphere(u1, u2);
	if (dot(dir, *Ns) < 0.)
		dir *= -1.f;
	*ray = Ray(org, dir);
	*pdf = shapeSet.Pdf(org) * INV_TWOPI;
	Spectrum Ls = L(org, *Ns, dir);
	return Ls;
}

Spectrum e_DiffuseLight::Sample_L(const e_KernelDynamicScene& scene, const float3& p, const LightSample& ls, float* pdf, e_VisibilitySegment* seg) const
{
	float3 ns;
	float3 ps = shapeSet.Sample(p, ls, &ns, scene.m_sBVHIntData.Data);
	seg->SetSegment(p, 0, ps, 0);
	*pdf = shapeSet.Pdf(p, seg->r.direction, scene.m_sBVHIntData.Data);
	return L(ps, ns, -seg->r.direction);
}

Spectrum e_DistantLight::Sample_L(const e_KernelDynamicScene& scene, const LightSample &ls, float u1, float u2, Ray *ray, float3 *Ns, float *pdf) const
{
	float3 worldCenter = scene.m_sBox.Center();
	float worldRadius = Distance(scene.m_sBox.maxV, scene.m_sBox.minV) / 2.0f;
	float d1, d2;
	ConcentricSampleDisk(ls.uPos[0], ls.uPos[1], &d1, &d2);
	float3 Pdisk = worldCenter + worldRadius * (d1 * sys.s + d2 * sys.t);
	*ray = Ray(Pdisk + worldRadius * lightDir, -1.0f * lightDir);
	*Ns = -1.0f * lightDir;
	*pdf = 1.f / (PI * worldRadius * worldRadius);
	return _L;
}

Spectrum e_DistantLight::Power(const e_KernelDynamicScene& scene) const
{
	float3 worldCenter = scene.m_sBox.Center();
	float worldRadius = Distance(scene.m_sBox.maxV, scene.m_sBox.minV) / 2.0f;
	return _L * PI * worldRadius * worldRadius;
}

Spectrum e_SpotLight::Sample_L(const e_KernelDynamicScene& scene, const LightSample &ls, float u1, float u2, Ray *ray, float3 *Ns, float *pdf) const
{
	float3 v = UniformSampleCone(ls.uPos[0], ls.uPos[1], cosTotalWidth);
	*ray = Ray(lightPos, sys.toWorld(v));
	*Ns = ray->direction;
	*pdf = UniformConePdf(cosTotalWidth);
	return Intensity * Falloff(v);
}

Spectrum e_SpotLight::Sample_L(const e_KernelDynamicScene& scene, const float3& p, const LightSample& ls, float* pdf, e_VisibilitySegment* seg) const
{
	seg->SetSegment(p, 0, lightPos, 0);
	*pdf = 1.0f;
	return Intensity * Falloff(sys.toLocal(-seg->r.direction)) / DistanceSquared(lightPos, p);
}

float e_SpotLight::Falloff(const float3 &w) const
{
	float3 wl = normalize(w);
	float costheta = wl.z;
	if (costheta < cosTotalWidth)     return 0.;
	if (costheta > cosFalloffStart)   return 1.;
	// Compute falloff inside spotlight cone
	float delta = (costheta - cosTotalWidth) / (cosFalloffStart - cosTotalWidth);
	return delta*delta*delta*delta;
}

e_InfiniteLight::e_InfiniteLight(const Spectrum& power, e_BufferReference<Distribution2D<4096, 4096>, Distribution2D<4096, 4096>>& d, e_BufferReference<e_MIPMap, e_KernelMIPMap>& mip)
	: e_LightBase(false), pDist(0)
{
	radianceMap = mip->getKernelData();
	void* pD = radianceMap.m_pDeviceData, *pH = malloc(mip->getBufferSize());
	cudaMemcpy(pH, pD, mip->getBufferSize(), cudaMemcpyDeviceToHost);
	radianceMap.m_pDeviceData = pH;
	unsigned int width = radianceMap.m_uWidth, height = radianceMap.m_uHeight;
	float filter = 1.0f / (float)MAX(width, height);
	float *img = new float[width*height];//I HATE new
	for (unsigned int v = 0; v < height; ++v)
	{
		float vp = (float)v / (float)height;
		float sinTheta = sinf(PI * float(v+.5f)/float(height));
		for (unsigned int u = 0; u < width; ++u)
		{
			float up = (float)u / (float)width;
			img[u+v*width] = radianceMap.Sample<Spectrum>(make_float2(up, vp), filter).getLuminance();
			img[u+v*width] *= sinTheta;
		}
	}
	pDist = d.getDevice();
	Distribution2D<4096, 4096>* t = d.operator()();
	t->Initialize(img, width, height);
	delete [] img;
	d.Invalidate();
	free(pH);
	radianceMap.m_pDeviceData = pD;
}

Spectrum e_InfiniteLight::Power(const e_KernelDynamicScene& scene) const
{
	float r = length(scene.m_sBox.Size()) / 2.0f;
	return PI * r * r * radianceMap.Sample<Spectrum>(make_float2(0.5f, 0.5f), 0.5f);
}

float e_InfiniteLight::Pdf(const e_KernelDynamicScene& scene, const float3 &p, const float3 &wi) const
{
	float theta = SphericalTheta(wi), phi = SphericalPhi(wi);
	float sintheta = sinf(theta);
	if (sintheta == 0.f)
		return 0.f;
	float p2 = pDist->Pdf(phi * INV_TWOPI, theta * INV_PI) / (2.f * PI * PI * sintheta);
	return p2;
}

Spectrum e_InfiniteLight::Sample_L(const e_KernelDynamicScene& scene, const LightSample &ls, float u1, float u2, Ray *ray, float3 *Ns, float *pdf) const
{
	float uv[2], mapPdf;
	pDist->SampleContinuous(ls.uPos[0], ls.uPos[1], uv, &mapPdf);
	if (mapPdf == 0.f)
		return Spectrum(0.0f);
	float theta = uv[1] * PI, phi = uv[0] * 2.f * PI;
	float costheta = cosf(theta), sintheta = sinf(theta);
	float sinphi = sinf(phi), cosphi = cosf(phi);
	float3 d = -make_float3(sintheta * cosphi, sintheta * sinphi, costheta);
	*Ns = d;
	float3 worldCenter = scene.m_sBox.Center();
	float worldRadius = length(scene.m_sBox.Size()) / 2.0f;
	Frame sys(d);
	float d1, d2;
	ConcentricSampleDisk(u1, u2, &d1, &d2);
	float3 Pdisk = worldCenter + worldRadius * (d1 * sys.t + d2 * sys.s);
	*ray = Ray(Pdisk + worldRadius * -d, d);
	float directionPdf = mapPdf / (2.f * PI * PI * sintheta);
	float areaPdf = 1.f / (PI * worldRadius * worldRadius);
	*pdf = directionPdf * areaPdf;
	if (sintheta == 0.f)
		*pdf = 0.f;
	return radianceMap.Sample<Spectrum>(make_float2(uv[0], uv[1]), 0);
}

Spectrum e_InfiniteLight::Sample_L(const e_KernelDynamicScene& scene, const float3& p, const LightSample& ls, float* pdf, e_VisibilitySegment* seg) const
{
	float uv[2], mapPdf;
	pDist->SampleContinuous(ls.uPos[0], ls.uPos[1], uv, &mapPdf);
	if (mapPdf == 0.0f)
		return Spectrum(0.0f);
	float theta = uv[1] * PI, phi = uv[0] * 2.f * PI;
	float costheta = cosf(theta), sintheta = sinf(theta);
	float sinphi = sinf(phi), cosphi = cosf(phi);
	float3 wi = make_float3(sintheta * cosphi, sintheta * sinphi, costheta);
	*pdf = mapPdf / (2.f * PI * PI * sintheta);
	if (sintheta == 0.f)
		*pdf = 0.f;
	seg->SetRay(p, 0, wi);
	return radianceMap.Sample<Spectrum>(make_float2(uv[0], uv[1]), 0);
}