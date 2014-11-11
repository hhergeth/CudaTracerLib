#include "e_Material.h"

e_KernelMaterial::e_KernelMaterial(const char* name)
{
	memset(Name, 0, sizeof(Name));
	if(name)
		memcpy(Name, name, strlen(name));
	HeightScale = 1.0f;
	NodeLightIndex = 0xffffffff;
	m_fAlphaThreshold = 1.0f;
	bsdf.type = 0;
	usedBssrdf = false;
}

CUDA_FUNC_IN float3 nor(float* D, int l, int t, int m, int r, int b, float HeightScale)
{
	//return normalize(cross(make_float3(0, -1, D[t] - D[m]), make_float3(-1, 0, D[l] - D[m])) + cross(make_float3(0, 1, D[b] - D[m]), make_float3(1, 0, D[r] - D[m])));
	return normalize(make_float3(D[m]-D[l], D[m]-D[t], HeightScale));
}

bool e_KernelMaterial::SampleNormalMap(DifferentialGeometry& dg) const
{
	if(NormalMap.used)
	{
		float3 n;
		NormalMap.tex.Evaluate(dg).toLinearRGB(n.x, n.y, n.z);
		float3 nWorld = dg.toWorld(n - make_float3(0.5f));
		dg.sys.n = normalize(nWorld);
		dg.sys.t = normalize(cross(nWorld, dg.sys.s));
		dg.sys.s = normalize(cross(nWorld, dg.sys.t));
		return true;
	}
	else if (HeightMap.used && HeightMap.tex.Is<e_KernelImageTexture>())
	{
		Spectrum grad[2];
		float2 uv = HeightMap.tex.As<e_KernelImageTexture>()->mapping.Map(dg);
		HeightMap.tex.As<e_KernelImageTexture>()->tex->evalGradient(uv, grad);
		float dDispDu = grad[0].getLuminance();
		float dDispDv = grad[1].getLuminance();
		float3 dpdu = dg.dpdu + dg.sys.n * (
			dDispDu - dot(dg.sys.n, dg.dpdu));
		float3 dpdv = dg.dpdv + dg.sys.n * (
			dDispDv - dot(dg.sys.n, dg.dpdv));

		dg.sys.n = normalize(cross(dpdu, dpdv));
		dg.sys.s = normalize(dpdu - dg.sys.n
			* dot(dg.sys.n, dpdu));
		dg.sys.t = cross(dg.sys.n, dg.sys.s);

		if (dot(dg.sys.n, dg.n) < 0)
			dg.sys.n *= -1;

		return true;
	}
	else return false;
}

float e_KernelMaterial::SampleAlphaMap(const DifferentialGeometry& uv) const
{
	if(AlphaMap.used)
	{//return 1;
		if(AlphaMap.tex.type == e_KernelImageTexture_TYPE)
		{
			float2 uv2 = AlphaMap.tex.As<e_KernelImageTexture>()->mapping.Map(uv);
			return AlphaMap.tex.As<e_KernelImageTexture>()->tex->SampleAlpha(uv2) != 1 ? 0 : 1;
		}
		Spectrum s = AlphaMap.tex.Evaluate(uv);
		if(s.isZero())
			return 0.0f;
		else return 1.0f;
	}
	else return 1.0f;
}

bool e_KernelMaterial::GetBSSRDF(const DifferentialGeometry& uv, const e_KernelBSSRDF** res) const
{
	*res = &bssrdf;
	return usedBssrdf;
}

void e_KernelMaterial::setBssrdf(const Spectrum& sig_a, const Spectrum& sigp_s, float e)
{
	usedBssrdf = true;
	bssrdf.e = e;
	bssrdf.sig_a = sig_a;
	bssrdf.sigp_s = sigp_s;
}
