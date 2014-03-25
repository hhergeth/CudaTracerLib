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

bool e_KernelMaterial::SampleNormalMap(const MapParameters& uv, float3* normal) const
{
	if(NormalMap.used)
	{
		float3 n;
		NormalMap.tex.Evaluate(uv).toLinearRGB(n.x,n.y,n.z);
		*normal = n * 2.0f - make_float3(1);
		return true;
	}
	else if(HeightMap.used)
	{
		float d = 1.0f / 256;//fucked up guess
		float m[16];
		for(int i = 0; i < 4; i++)
			for(int j = 0; j < 4; j++)
			{
				MapParameters mp = uv;
				*(float2*)&mp.uv = mp.uv + make_float2(i - 1, j - 1) * d;
				m[i * 4 + j] = HeightMap.tex.Evaluate(mp).average();
			}
		*normal = nor(m, 4, 1, 5, 6, 9, HeightScale); 
		return true;
	}
	else return false;
}

float e_KernelMaterial::SampleAlphaMap(const MapParameters& uv) const
{
	if(AlphaMap.used)
	{//return 1;
		if(AlphaMap.tex.type == e_KernelImageTexture_TYPE)
		{
			return AlphaMap.tex.As<e_KernelImageTexture>()->tex->SampleAlpha(uv.uv) != 1 ? 0 : 1;
		}
		Spectrum s = AlphaMap.tex.Evaluate(uv);
		if(s.isZero())
			return 0.0f;
		else return 1.0f;
	}
	else return 1.0f;
}

bool e_KernelMaterial::GetBSSRDF(const MapParameters& uv, const e_KernelBSSRDF** res) const
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
