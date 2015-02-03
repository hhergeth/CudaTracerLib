#include "e_Material.h"

e_KernelMaterial::e_KernelMaterial(const char* name)
{
	enableParallaxOcclusion = false;
	parallaxMinSamples = 10;
	parallaxMaxSamples = 50;
	memset(Name, 0, sizeof(Name));
	if(name)
		memcpy(Name, name, strlen(name));
	HeightScale = 1.0f;
	NodeLightIndex = 0xffffffff;
	m_fAlphaThreshold = 1.0f;
	bsdf.type = 0;
	usedBssrdf = false;
}

CUDA_FUNC_IN float2 parallaxOcclusion(const float2& texCoord, e_KernelMIPMap* tex, const float3& vViewTS, float HeightScale, int MinSamples, int MaxSamples)
{
	const float2 vParallaxDirection = normalize(!vViewTS);
	float fLength = length(vViewTS);
	float fParallaxLength = sqrt(fLength * fLength - vViewTS.z * vViewTS.z) / vViewTS.z;
	const float2 vParallaxOffsetTS = vParallaxDirection * fParallaxLength * HeightScale;

	int nNumSteps = (int)lerp(MaxSamples, MinSamples, Frame::cosTheta(normalize(vViewTS)));
	float CurrHeight = 0.0f;
	float StepSize = 1.0 / (float)nNumSteps;
	float PrevHeight = 1.0;
	int    StepIndex = 0;
	float2 TexOffsetPerStep = StepSize * vParallaxOffsetTS;
	float2 TexCurrentOffset = texCoord;
	float  CurrentBound = 1.0;
	float  ParallaxAmount = 0.0;

	// Create the 2D vectors that will be used as the sampled points 
	float2 pt1 = make_float2(0);
	float2 pt2 = make_float2(0);

	float2 texOffset2 = make_float2(0);

	while (StepIndex < nNumSteps)
	{
		// Backtrace along the parallax offset vector by the StepSize value
		TexCurrentOffset -= TexOffsetPerStep;

		// Sample height map which in this case is stored in the alpha channel of the normal map:
		CurrHeight = tex->Sample(TexCurrentOffset).average();

		// Decrease the current height of the eye vector by the StepSize value
		CurrentBound -= StepSize;

		/*
		Check if the eye vector has a larger height value than the CurrHeight sampled from the
		height map which would mean there has been no intersection yet so increment the StepIndex.
		If there has been an intersection (eye vector has a smaller value than CurrHeight) it will lie
		between pt1 and pt2 which are the current sample point and the previous sampled point.
		*/
		if (CurrHeight > CurrentBound)
		{
			pt1 = make_float2(CurrentBound, CurrHeight);
			pt2 = make_float2(CurrentBound + StepSize, PrevHeight);

			texOffset2 = TexCurrentOffset - TexOffsetPerStep;

			StepIndex = nNumSteps + 1;
			PrevHeight = CurrHeight;
		}
		else
		{
			StepIndex++;
			PrevHeight = CurrHeight;
		}
	}
	float Delta2 = pt2.x - pt2.y;
	float Delta1 = pt1.x - pt1.y;
	float Denominator = Delta2 - Delta1;
	ParallaxAmount = Denominator != 0 ? (pt1.x * Delta2 - pt2.x * Delta1) / Denominator : 0;
	float2 ParallaxOffset = vParallaxOffsetTS * (1 - ParallaxAmount);
	return texCoord - ParallaxOffset;
}

bool e_KernelMaterial::SampleNormalMap(DifferentialGeometry& dg, const float3& wi) const
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
		e_KernelTextureMapping2D& map = HeightMap.tex.As<e_KernelImageTexture>()->mapping;
		float2 uv = map.Map(dg);
		if (enableParallaxOcclusion)
		{
			uv = parallaxOcclusion(uv, HeightMap.tex.As<e_KernelImageTexture>()->tex.operator->(), dg.toLocal(-wi), HeightScale, parallaxMinSamples, parallaxMaxSamples);
			dg.uv[0] = (uv - make_float2(map.du, map.dv)) / make_float2(map.su, map.sv);
		}

		Spectrum grad[2];
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
	if (usedBssrdf)
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
