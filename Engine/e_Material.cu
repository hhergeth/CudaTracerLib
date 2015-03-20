#include "e_Material.h"

e_KernelMaterial::e_KernelMaterial()
{
	parallaxMinSamples = 10;
	parallaxMaxSamples = 50;
	enableParallaxOcclusion = false;
	Name = "NoNameMaterial";
	HeightScale = 1.0f;
	NodeLightIndex = 0xffffffff;
	m_fAlphaThreshold = 1.0f;
	bsdf.setTypeToken(0);
	usedBssrdf = 0;
	AlphaMap.used = NormalMap.used = HeightMap.used = 0;
}

e_KernelMaterial::e_KernelMaterial(const char* name)
{
	parallaxMinSamples = 10;
	parallaxMaxSamples = 50;
	enableParallaxOcclusion = false;
	Name = name;
	HeightScale = 1.0f;
	NodeLightIndex = 0xffffffff;
	m_fAlphaThreshold = 1.0f;
	bsdf.setTypeToken(0);
	usedBssrdf = 0;
	AlphaMap.used = NormalMap.used = HeightMap.used = 0;
}

CUDA_FUNC_IN Vec2f parallaxOcclusion(const Vec2f& texCoord, e_KernelMIPMap* tex, const Vec3f& vViewTS, float HeightScale, int MinSamples, int MaxSamples)
{
	const Vec2f vParallaxDirection = normalize(vViewTS.getXY());
	float fLength = length(vViewTS);
	float fParallaxLength = sqrt(fLength * fLength - vViewTS.z * vViewTS.z) / vViewTS.z;
	const Vec2f vParallaxOffsetTS = vParallaxDirection * fParallaxLength * HeightScale;

	int nNumSteps = (int)math::lerp(MaxSamples, MinSamples, Frame::cosTheta(normalize(vViewTS)));
	float CurrHeight = 0.0f;
	float StepSize = 1.0 / (float)nNumSteps;
	float PrevHeight = 1.0;
	int    StepIndex = 0;
	Vec2f TexOffsetPerStep = StepSize * vParallaxOffsetTS;
	Vec2f TexCurrentOffset = texCoord;
	float  CurrentBound = 1.0;
	float  ParallaxAmount = 0.0;

	// Create the 2D vectors that will be used as the sampled points 
	Vec2f pt1 = Vec2f(0);
	Vec2f pt2 = Vec2f(0);

	Vec2f texOffset2 = Vec2f(0);

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
			pt1 = Vec2f(CurrentBound, CurrHeight);
			pt2 = Vec2f(CurrentBound + StepSize, PrevHeight);

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
	Vec2f ParallaxOffset = vParallaxOffsetTS * (1 - ParallaxAmount);
	return texCoord - ParallaxOffset;
}

bool e_KernelMaterial::SampleNormalMap(DifferentialGeometry& dg, const Vec3f& wi) const
{
	if(NormalMap.used)
	{
		Vec3f n;
		NormalMap.tex.Evaluate(dg).toLinearRGB(n.x, n.y, n.z);
		Vec3f nWorld = dg.toWorld(n - Vec3f(0.5f));
		dg.sys.n = normalize(nWorld);
		dg.sys.t = normalize(cross(nWorld, dg.sys.s));
		dg.sys.s = normalize(cross(nWorld, dg.sys.t));
		return true;
	}
	else if (HeightMap.used && HeightMap.tex.Is<e_ImageTexture>())
	{
		e_TextureMapping2D& map = HeightMap.tex.As<e_ImageTexture>()->mapping;
		Vec2f uv = map.Map(dg);
		if (enableParallaxOcclusion)
		{
			uv = parallaxOcclusion(uv, HeightMap.tex.As<e_ImageTexture>()->tex.operator->(), dg.toLocal(-wi), HeightScale, parallaxMinSamples, parallaxMaxSamples);
			dg.uv[0] = (uv - Vec2f(map.du, map.dv)) / Vec2f(map.su, map.sv);
		}

		Spectrum grad[2];
		HeightMap.tex.As<e_ImageTexture>()->tex->evalGradient(uv, grad);
		float dDispDu = grad[0].getLuminance();
		float dDispDv = grad[1].getLuminance();
		Vec3f dpdu = dg.dpdu + dg.sys.n * (
			dDispDu - dot(dg.sys.n, dg.dpdu));
		Vec3f dpdv = dg.dpdv + dg.sys.n * (
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
		if (AlphaMap.tex.Is<e_ImageTexture>())
		{
			Vec2f uv2 = AlphaMap.tex.As<e_ImageTexture>()->mapping.Map(uv);
			return AlphaMap.tex.As<e_ImageTexture>()->tex->SampleAlpha(uv2) != 1 ? 0 : 1;
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
	usedBssrdf = 1;
	bssrdf.e = e;
	bssrdf.sig_a = sig_a;
	bssrdf.sigp_s = sigp_s;
}
